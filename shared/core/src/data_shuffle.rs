use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
#[repr(C)]
pub enum Shuffle {
    DontShuffle,
    Seeded([u8; 32]),
}
