//! Reproducible SQL-file client preprocessing benchmark (no database access).
//!
//! Run the SAME example/input/profile against the old and new source:
//! cargo run -p dbx-sql-core --release --example sql_file_import_bench -- dump.sql 7
//!
//! This measures MySQL splitting, controlled batch planning and one progress
//! summary per planned statement. Input loading/UTF-8 validation are outside the
//! timer. The production decoder, database execution, transactions, IPC and UI
//! are not measured. A full-output checksum and allocation accounting each run
//! separately, outside the timed samples. Allocated bytes are cumulative, not
//! peak/live memory. No timing assertion is used as a CI correctness test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dbx_sql_core::models::connection::DatabaseType;
use dbx_sql_core::sql::{
    optimize_sql_file_import_statements, statement_summary, SqlFileImportStatementKind, SqlParsingOptions,
    SqlStatementSplitter, SqlStatementWithControl,
};

const CHUNK_BYTES: usize = 256 * 1024;
const BATCH_STATEMENTS: usize = 256;
const BATCH_BYTES: usize = 8 * 1024 * 1024;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

struct CountingAllocator;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

// The wrapper always forwards the original pointer and layout to System.
// Only the separate allocation sample enables the counters.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Debug, PartialEq, Eq, serde::Serialize)]
struct Stats {
    source_statements: usize,
    planned_statements: usize,
    skipped_statements: usize,
    source_sql_bytes: usize,
    prepared_sql_bytes: usize,
    summary_bytes: usize,
    summary_checksum_fnv1a64: u64,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            source_statements: 0,
            planned_statements: 0,
            skipped_statements: 0,
            source_sql_bytes: 0,
            prepared_sql_bytes: 0,
            summary_bytes: 0,
            summary_checksum_fnv1a64: FNV_OFFSET,
        }
    }
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
}

// Include length prefixes so different statement boundaries do not hash as the
// same concatenated text. This is an output comparison fingerprint, not crypto.
fn hash_field(hash: &mut u64, bytes: &[u8]) {
    hash_bytes(hash, &(bytes.len() as u64).to_le_bytes());
    hash_bytes(hash, bytes);
}

fn flush_batch(pending: &mut Vec<SqlStatementWithControl>, stats: &mut Stats, sql_hash: Option<&mut u64>) {
    if pending.is_empty() {
        return;
    }
    let statements = std::mem::take(pending);
    stats.source_statements += statements.len();
    stats.source_sql_bytes += statements.iter().map(|statement| statement.sql.len()).sum::<usize>();

    // Match optimize_controlled_sql_file_import_statements: each contiguous
    // stop_on_error group is cloned and planned separately, in source order.
    let mut planned = Vec::new();
    let mut start = 0;
    while start < statements.len() {
        let stop_on_error = statements[start].stop_on_error;
        let mut end = start + 1;
        while end < statements.len() && statements[end].stop_on_error == stop_on_error {
            end += 1;
        }
        let sql = statements[start..end].iter().map(|statement| statement.sql.clone()).collect::<Vec<_>>();
        planned.extend(
            optimize_sql_file_import_statements(&sql, Some(DatabaseType::Mysql), None)
                .into_iter()
                .map(|statement| (statement, stop_on_error)),
        );
        start = end;
    }

    let mut sql_hash = sql_hash;
    for (statement, stop_on_error) in planned {
        stats.planned_statements += 1;
        let skipped = statement.kind == SqlFileImportStatementKind::Skip;
        stats.skipped_statements += usize::from(skipped);
        stats.prepared_sql_bytes += statement.sql.len();
        // The success path in execute_statement_with_progress calls this once.
        let summary = statement_summary(black_box(&statement.sql));
        stats.summary_bytes += summary.len();
        hash_field(&mut stats.summary_checksum_fnv1a64, summary.as_bytes());
        if let Some(hash) = sql_hash.as_deref_mut() {
            hash_field(hash, statement.sql.as_bytes());
            hash_field(hash, &[u8::from(skipped), u8::from(stop_on_error)]);
            hash_field(hash, &(statement.source_statement_count as u64).to_le_bytes());
        }
        black_box(statement);
    }
}

fn preprocess(input: &str, full_checksum: bool) -> (Stats, u64) {
    let mut splitter = SqlStatementSplitter::with_options(SqlParsingOptions::for_database_type(DatabaseType::Mysql));
    let mut pending = Vec::with_capacity(BATCH_STATEMENTS);
    let mut stats = Stats::default();
    let mut sql_hash = FNV_OFFSET;
    let mut start = 0;
    while start < input.len() {
        let mut end = (start + CHUNK_BYTES).min(input.len());
        // Input was validated once before timing. Preserve characters spanning
        // the nominal byte boundary without allocating a decoded chunk copy.
        while !input.is_char_boundary(end) {
            end -= 1;
        }
        pending.extend(splitter.push_chunk_with_control(&input[start..end]));
        // The production importer checks AFTER all statements from a chunk are
        // appended, rather than flushing individually at exactly 256 statements.
        let buffered_bytes: usize = pending.iter().map(|statement| statement.sql.len()).sum();
        if pending.len() >= BATCH_STATEMENTS || buffered_bytes >= BATCH_BYTES {
            flush_batch(&mut pending, &mut stats, full_checksum.then_some(&mut sql_hash));
        }
        start = end;
    }
    pending.extend(splitter.finish_with_control());
    flush_batch(&mut pending, &mut stats, full_checksum.then_some(&mut sql_hash));
    (stats, sql_hash)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(args.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "Usage: sql_file_import_bench <UTF-8.sql> [runs=7]")
    })?);
    let runs: usize = args.next().map(|value| value.to_string_lossy().parse()).transpose()?.unwrap_or(7);
    if runs == 0 || runs > 100 || args.next().is_some() {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "runs must be 1..100; only one SQL path is accepted").into()
        );
    }
    let input = std::fs::read_to_string(&path)?;
    let input = input.strip_prefix('\u{feff}').unwrap_or(&input);

    let (expected, sql_hash) = preprocess(black_box(input), true); // Warm-up and untimed output fingerprint.
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let (stats, _) = black_box(preprocess(black_box(input), false));
        let elapsed = start.elapsed();
        assert_eq!(stats, expected, "Preprocessing output changed between samples");
        samples.push(elapsed);
    }

    ALLOC_CALLS.store(0, Ordering::Relaxed);
    REALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    TRACK_ALLOCATIONS.store(true, Ordering::Relaxed);
    let (allocation_stats, _) = preprocess(black_box(input), false);
    TRACK_ALLOCATIONS.store(false, Ordering::Relaxed);
    assert_eq!(allocation_stats, expected);

    let milliseconds = |value: Duration| value.as_secs_f64() * 1000.0;
    let sample_ms = samples.iter().map(|&value| milliseconds(value)).collect::<Vec<_>>();
    samples.sort_unstable();
    let median_ms = if runs.is_multiple_of(2) {
        (milliseconds(samples[runs / 2 - 1]) + milliseconds(samples[runs / 2])) / 2.0
    } else {
        milliseconds(samples[runs / 2])
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "scope": "MySQL split/plan/summary CPU preprocessing; NOT end-to-end import",
            "file": path,
            "utf8_bytes": input.len(),
            "chunk_bytes": CHUNK_BYTES,
            "batch_statement_threshold": BATCH_STATEMENTS,
            "batch_byte_threshold": BATCH_BYTES,
            "timed_runs": runs,
            "samples_ms": sample_ms,
            "median_ms": median_ms,
            "minimum_ms": milliseconds(samples[0]),
            "maximum_ms": milliseconds(samples[runs - 1]),
            "stats": expected,
            "prepared_output_fnv1a64": format!("{sql_hash:016x}"),
            "allocation_pass": {
                "alloc_calls": ALLOC_CALLS.load(Ordering::Relaxed),
                "realloc_calls": REALLOC_CALLS.load(Ordering::Relaxed),
                "cumulative_requested_bytes": ALLOCATED_BYTES.load(Ordering::Relaxed),
                "note": "Separate untimed pass; cumulative requests include reallocations, not peak/live memory"
            },
            "database_execution": "NOT_RUN",
            "data_rows": null,
            "excluded": ["disk IO", "production encoding decoder", "database/network", "IPC/UI", "full SQL fingerprint", "allocation counting"],
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_boundary_and_statement_counts_are_preserved() {
        let prefix = "INSERT INTO t VALUES ('";
        let mut input = format!("{prefix}{}", "x".repeat(CHUNK_BYTES - prefix.len() - 1));
        // The four-byte emoji begins one byte before the nominal chunk edge.
        input.push_str("😀中文');\nINSERT INTO t VALUES ('second');");
        let (stats, _) = preprocess(&input, true);
        assert_eq!(stats.source_statements, 2);
        assert_eq!(stats.planned_statements, 2);
        assert_eq!(stats.skipped_statements, 0);
    }

    #[test]
    fn empty_file_has_no_statements() {
        let (stats, hash) = preprocess("", true);
        assert_eq!(stats, Stats::default());
        assert_eq!(hash, FNV_OFFSET);
    }
}
