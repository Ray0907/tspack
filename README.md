# tspack

Schema-driven IoT time-series data encoding for resource-constrained environments.

## Why tspack?

Every IoT device should transmit maximum information with minimum energy.

```
Problem: JSON uses 54 bytes for 3 sensor values
         MessagePack uses 16 bytes
         tspack uses 7 bytes (87% smaller than JSON)
```

## Features

- **Schema-driven**: Use known value ranges to achieve theoretical compression limits
- **Correctness-first**: CRC-8 integrity check, quality flags (Valid/Null/Overflow/Underflow)
- **Streaming**: Delta encoding with checkpoints for time-series data
- **no_std**: Zero dependencies, embedded-friendly
- **Fast**: 92 M/s encode throughput, 2.6x faster than MessagePack

## Benchmark Results

Tested with real solar power IoT data (Kaggle, 3182 samples):

| Format | Size | vs tspack |
|--------|------|-----------|
| tspack delta | 19 KB | 1.0x |
| tspack absolute | 22 KB | 1.2x |
| MessagePack | 181 KB | **9.5x** |
| JSON | 539 KB | **28.2x** |

Encoding speed (single point):

| Format | Time | Throughput |
|--------|------|------------|
| tspack | 10.8 ns | 92.5 M/s |
| MessagePack | 28.4 ns | 35.1 M/s |
| Gorilla | 60.2 ns | 16.6 M/s |
| JSON | 97.1 ns | 10.3 M/s |

## Quick Start

```rust
use tspack::{Schema, Field, encode, decode, EncodeOptions};

// Define schema with value ranges and precision
let schema = Schema::new(1, &[
    Field::new(-40.0, 80.0, 0.1),   // temperature
    Field::new(0.0, 100.0, 1.0),    // humidity
    Field::new(300.0, 1200.0, 0.1), // pressure
]);

// Encode
let values = [Some(23.5), Some(65.0), Some(1013.2)];
let mut buf = [0u8; 32];
let result = encode(&schema, &values, &mut buf, EncodeOptions::default())?;
// result.len == 7 bytes

// Decode
let decoded = decode(&schema, &buf[..result.len])?;
```

## Streaming with Delta Encoding

```rust
use tspack::{DeltaSchema, Field, StreamEncoder};

let schema = DeltaSchema::new(1, &[
    (Field::new(-40.0, 80.0, 0.1), 2.0),  // max change per sample
    (Field::new(0.0, 100.0, 1.0), 5.0),
], 32); // checkpoint every 32 samples

let mut encoder = StreamEncoder::new();
let mut buf = [0u8; 32];

// First sample: absolute frame
let (len, is_abs) = encoder.encode(&schema, &values, &mut buf)?;

// Subsequent samples: delta frames (smaller)
let (len, is_abs) = encoder.encode(&schema, &next_values, &mut buf)?;
```

## Installation

```toml
[dependencies]
tspack = "0.1"
```

## Documentation

See [plans.md](plans.md) for detailed design document including:
- Information theory background
- Wire format specification
- Algorithm details (QDC: Quantized Delta with Checkpoint)
- Benchmark methodology and results

## Roadmap

- [x] Core encoding/decoding
- [x] Delta streaming encoder
- [x] CRC-8 integrity check
- [x] Benchmark vs MessagePack/JSON/Gorilla
- [ ] C bindings (cbindgen)
- [ ] Python bindings (PyO3)
- [ ] ESP32/Arduino examples
- [ ] MQTT integration
- [ ] Schema registry

## License

MIT
