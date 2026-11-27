//! MVS: what the player is doing right now, so the web allocator can tag large allocations with it
//! and the memory report can say which part of the player holds them.

use std::sync::atomic::{AtomicU8, Ordering::Relaxed};

pub const OTHER: u8 = 0;
pub const RENDER: u8 = 1;
pub const SCRIPT: u8 = 2;
pub const DECODE: u8 = 3;
pub const LOAD: u8 = 4;
pub const TESSELLATE: u8 = 5;
pub const COUNT: usize = 6;
pub const NAMES: [&str; COUNT] = ["other", "render", "script", "decode", "load", "tessellate"];

static PHASE: AtomicU8 = AtomicU8::new(OTHER);

/// Restores the previous phase when dropped.
pub struct Guard(u8);

impl Drop for Guard {
    fn drop(&mut self) {
        PHASE.store(self.0, Relaxed);
    }
}

#[must_use]
pub fn enter(phase: u8) -> Guard {
    Guard(PHASE.swap(phase, Relaxed))
}

pub fn current() -> u8 {
    PHASE.load(Relaxed)
}
