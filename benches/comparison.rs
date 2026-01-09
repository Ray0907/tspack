//! Benchmark comparison of tspack vs other encoding formats

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tspack::{Schema, Field, encode, decode, EncodeOptions};
use tsz::{DataPoint, Encode, Decode, StdEncoder, StdDecoder};
use serde::{Serialize, Deserialize};
use std::path::Path;

/// Test data point for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SensorData {
    temperature: f32,
    humidity: f32,
    pressure: f32,
}

/// Generate test data with different characteristics
fn generate_smooth_data(count: usize) -> Vec<SensorData> {
    (0..count)
        .map(|i| {
            let t = i as f32 * 0.1;
            SensorData {
                temperature: 20.0 + 5.0 * (t * 0.1).sin(),
                humidity: 50.0 + 10.0 * (t * 0.05).cos(),
                pressure: 1013.0 + 2.0 * (t * 0.02).sin(),
            }
        })
        .collect()
}

fn generate_noisy_data(count: usize) -> Vec<SensorData> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|i| {
            let t = i as f32 * 0.1;
            SensorData {
                temperature: 20.0 + 5.0 * (t * 0.1).sin() + rng.gen_range(-2.0..2.0),
                humidity: 50.0 + 10.0 * (t * 0.05).cos() + rng.gen_range(-5.0..5.0),
                pressure: 1013.0 + 2.0 * (t * 0.02).sin() + rng.gen_range(-1.0..1.0),
            }
        })
        .collect()
}

fn generate_random_data(count: usize) -> Vec<SensorData> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|_| SensorData {
            temperature: rng.gen_range(-40.0..80.0),
            humidity: rng.gen_range(0.0..100.0),
            pressure: rng.gen_range(300.0..1200.0),
        })
        .collect()
}

/// Benchmark encoding speed
fn bench_encode(c: &mut Criterion) {
    let schema = Schema::new(
        1,
        &[
            Field::new(-40.0, 80.0, 0.1),
            Field::new(0.0, 100.0, 1.0),
            Field::new(300.0, 1200.0, 0.1),
        ],
    );

    let smooth_data = generate_smooth_data(1000);
    let noisy_data = generate_noisy_data(1000);
    let random_data = generate_random_data(1000);

    let mut group = c.benchmark_group("encode_single");
    group.throughput(Throughput::Elements(1));

    // tspack - smooth data
    group.bench_function("tspack/smooth", |b| {
        let data = &smooth_data[0];
        let values = [Some(data.temperature), Some(data.humidity), Some(data.pressure)];
        let mut buf = [0u8; 32];
        b.iter(|| {
            buf.fill(0);
            encode(black_box(&schema), black_box(&values), &mut buf, EncodeOptions::default())
        })
    });

    // tspack - random data
    group.bench_function("tspack/random", |b| {
        let data = &random_data[0];
        let values = [Some(data.temperature), Some(data.humidity), Some(data.pressure)];
        let mut buf = [0u8; 32];
        b.iter(|| {
            buf.fill(0);
            encode(black_box(&schema), black_box(&values), &mut buf, EncodeOptions::default())
        })
    });

    // JSON
    group.bench_function("json/smooth", |b| {
        let data = &smooth_data[0];
        b.iter(|| serde_json::to_vec(black_box(data)))
    });

    group.bench_function("json/random", |b| {
        let data = &random_data[0];
        b.iter(|| serde_json::to_vec(black_box(data)))
    });

    // MessagePack
    group.bench_function("msgpack/smooth", |b| {
        let data = &smooth_data[0];
        b.iter(|| rmp_serde::to_vec(black_box(data)))
    });

    group.bench_function("msgpack/random", |b| {
        let data = &random_data[0];
        b.iter(|| rmp_serde::to_vec(black_box(data)))
    });

    // Gorilla (tsz) - single value
    group.bench_function("gorilla/smooth", |b| {
        let data = &smooth_data[0];
        b.iter(|| {
            let w = tsz::stream::BufferedWriter::new();
            let mut encoder = StdEncoder::new(1000000000, w);
            let dp = DataPoint::new(1000000000, data.temperature as f64);
            encoder.encode(black_box(dp))
        })
    });

    group.finish();
}

/// Benchmark decoding speed
fn bench_decode(c: &mut Criterion) {
    let schema = Schema::new(
        1,
        &[
            Field::new(-40.0, 80.0, 0.1),
            Field::new(0.0, 100.0, 1.0),
            Field::new(300.0, 1200.0, 0.1),
        ],
    );

    let smooth_data = generate_smooth_data(1);
    let data = &smooth_data[0];

    // Pre-encode data
    let values = [Some(data.temperature), Some(data.humidity), Some(data.pressure)];
    let mut tspack_buf = [0u8; 32];
    let tspack_result = encode(&schema, &values, &mut tspack_buf, EncodeOptions::default()).unwrap();
    let tspack_encoded = &tspack_buf[..tspack_result.len];

    let json_encoded = serde_json::to_vec(data).unwrap();
    let msgpack_encoded = rmp_serde::to_vec(data).unwrap();

    // Pre-encode Gorilla data
    let gorilla_encoded = {
        let w = tsz::stream::BufferedWriter::new();
        let mut encoder = StdEncoder::new(1000000000, w);
        let dp = DataPoint::new(1000000000, data.temperature as f64);
        encoder.encode(dp);
        encoder.close()
    };

    let mut group = c.benchmark_group("decode_single");
    group.throughput(Throughput::Elements(1));

    group.bench_function("tspack", |b| {
        b.iter(|| decode(black_box(&schema), black_box(tspack_encoded)))
    });

    group.bench_function("json", |b| {
        b.iter(|| serde_json::from_slice::<SensorData>(black_box(&json_encoded)))
    });

    group.bench_function("msgpack", |b| {
        b.iter(|| rmp_serde::from_slice::<SensorData>(black_box(&msgpack_encoded)))
    });

    // Gorilla decode
    group.bench_function("gorilla", |b| {
        b.iter(|| {
            let r = tsz::stream::BufferedReader::new(black_box(gorilla_encoded.clone()));
            let mut decoder = StdDecoder::new(r);
            decoder.next()
        })
    });

    group.finish();
}

/// Benchmark encoded size comparison
fn bench_size(c: &mut Criterion) {
    let schema = Schema::new(
        1,
        &[
            Field::new(-40.0, 80.0, 0.1),
            Field::new(0.0, 100.0, 1.0),
            Field::new(300.0, 1200.0, 0.1),
        ],
    );

    let data = SensorData {
        temperature: 23.5,
        humidity: 65.0,
        pressure: 1013.2,
    };

    // Calculate sizes
    let values = [Some(data.temperature), Some(data.humidity), Some(data.pressure)];
    let mut tspack_buf = [0u8; 32];
    let tspack_result = encode(&schema, &values, &mut tspack_buf, EncodeOptions::default()).unwrap();

    let json_encoded = serde_json::to_vec(&data).unwrap();
    let msgpack_encoded = rmp_serde::to_vec(&data).unwrap();

    // Gorilla encoded size (single value with timestamp)
    let gorilla_encoded = {
        let w = tsz::stream::BufferedWriter::new();
        let mut encoder = StdEncoder::new(1000000000, w);
        let dp = DataPoint::new(1000000000, data.temperature as f64);
        encoder.encode(dp);
        encoder.close()
    };

    println!("\n=== Encoded Size Comparison (3 sensor values) ===");
    println!("tspack:      {} bytes (bit-packed, no field names)", tspack_result.len);
    println!("JSON:        {} bytes", json_encoded.len());
    println!("MessagePack: {} bytes", msgpack_encoded.len());
    println!("Gorilla:     {} bytes (1 value only, includes timestamp)", gorilla_encoded.len());
    println!("=================================================\n");

    // Dummy benchmark just to ensure this runs
    let mut group = c.benchmark_group("size_comparison");
    group.bench_function("report", |b| b.iter(|| 1 + 1));
    group.finish();
}

/// Benchmark batch encoding
fn bench_batch(c: &mut Criterion) {
    let schema = Schema::new(
        1,
        &[
            Field::new(-40.0, 80.0, 0.1),
            Field::new(0.0, 100.0, 1.0),
            Field::new(300.0, 1200.0, 0.1),
        ],
    );

    let smooth_data = generate_smooth_data(100);

    let mut group = c.benchmark_group("encode_batch_100");
    group.throughput(Throughput::Elements(100));

    // tspack - encode 100 points
    group.bench_function("tspack", |b| {
        let mut buf = [0u8; 32];
        b.iter(|| {
            for data in &smooth_data {
                buf.fill(0);
                let values = [Some(data.temperature), Some(data.humidity), Some(data.pressure)];
                let _ = encode(black_box(&schema), black_box(&values), &mut buf, EncodeOptions::default());
            }
        })
    });

    // JSON - encode 100 points
    group.bench_function("json", |b| {
        b.iter(|| {
            for data in &smooth_data {
                let _ = serde_json::to_vec(black_box(data));
            }
        })
    });

    // MessagePack - encode 100 points
    group.bench_function("msgpack", |b| {
        b.iter(|| {
            for data in &smooth_data {
                let _ = rmp_serde::to_vec(black_box(data));
            }
        })
    });

    group.finish();
}

/// Benchmark delta encoding vs absolute
fn bench_delta(c: &mut Criterion) {
    use tspack::{DeltaSchema, StreamEncoder, StreamDecoder};

    let delta_schema = DeltaSchema::new(
        1,
        &[
            // Temperature: max change 2.0 per sample
            (Field::new(-40.0, 80.0, 0.1), 2.0),
            // Humidity: max change 5.0 per sample
            (Field::new(0.0, 100.0, 1.0), 5.0),
            // Pressure: max change 1.0 per sample
            (Field::new(300.0, 1200.0, 0.1), 1.0),
        ],
        32,
    );

    let absolute_schema = Schema::new(
        1,
        &[
            Field::new(-40.0, 80.0, 0.1),
            Field::new(0.0, 100.0, 1.0),
            Field::new(300.0, 1200.0, 0.1),
        ],
    );

    let smooth_data = generate_smooth_data(100);

    // Calculate compression ratios
    {
        let mut encoder = StreamEncoder::new();
        let mut buf = [0u8; 32];
        let mut total_delta_bytes = 0usize;
        let mut total_absolute_bytes = 0usize;
        let mut absolute_frames = 0;
        let mut delta_frames = 0;

        for data in &smooth_data {
            let values = [Some(data.temperature), Some(data.humidity), Some(data.pressure)];

            // Delta encoding
            let (len, is_abs) = encoder.encode(&delta_schema, &values, &mut buf).unwrap();
            total_delta_bytes += len;
            if is_abs {
                absolute_frames += 1;
            } else {
                delta_frames += 1;
            }

            // Absolute encoding
            buf.fill(0);
            let result = encode(&absolute_schema, &values, &mut buf, EncodeOptions::default()).unwrap();
            total_absolute_bytes += result.len;
        }

        let json_bytes: usize = smooth_data.iter()
            .map(|d| serde_json::to_vec(d).unwrap().len())
            .sum();

        let msgpack_bytes: usize = smooth_data.iter()
            .map(|d| rmp_serde::to_vec(d).unwrap().len())
            .sum();

        println!("\n=== Streaming Compression (100 samples) ===");
        println!("tspack delta:    {} bytes ({} abs + {} delta frames)",
                 total_delta_bytes, absolute_frames, delta_frames);
        println!("tspack absolute: {} bytes", total_absolute_bytes);
        println!("MessagePack:     {} bytes", msgpack_bytes);
        println!("JSON:            {} bytes", json_bytes);
        println!("Delta savings:   {:.1}% vs absolute",
                 (1.0 - total_delta_bytes as f64 / total_absolute_bytes as f64) * 100.0);
        println!("============================================\n");
    }

    let mut group = c.benchmark_group("encode_stream_100");
    group.throughput(Throughput::Elements(100));

    // Delta streaming encode
    group.bench_function("tspack_delta", |b| {
        let mut encoder = StreamEncoder::new();
        let mut buf = [0u8; 32];
        b.iter(|| {
            encoder.reset();
            for data in &smooth_data {
                let values = [Some(data.temperature), Some(data.humidity), Some(data.pressure)];
                let _ = encoder.encode(black_box(&delta_schema), black_box(&values), &mut buf);
            }
        })
    });

    // Absolute encode (existing)
    group.bench_function("tspack_absolute", |b| {
        let mut buf = [0u8; 32];
        b.iter(|| {
            for data in &smooth_data {
                buf.fill(0);
                let values = [Some(data.temperature), Some(data.humidity), Some(data.pressure)];
                let _ = encode(black_box(&absolute_schema), black_box(&values), &mut buf, EncodeOptions::default());
            }
        })
    });

    group.finish();

    // Decode benchmark
    let mut decode_group = c.benchmark_group("decode_stream_100");
    decode_group.throughput(Throughput::Elements(100));

    // Pre-encode delta frames
    let delta_frames: Vec<Vec<u8>> = {
        let mut encoder = StreamEncoder::new();
        let mut buf = [0u8; 32];
        smooth_data.iter().map(|data| {
            let values = [Some(data.temperature), Some(data.humidity), Some(data.pressure)];
            let (len, _) = encoder.encode(&delta_schema, &values, &mut buf).unwrap();
            buf[..len].to_vec()
        }).collect()
    };

    // Pre-encode absolute frames
    let absolute_frames: Vec<Vec<u8>> = {
        let mut buf = [0u8; 32];
        smooth_data.iter().map(|data| {
            let values = [Some(data.temperature), Some(data.humidity), Some(data.pressure)];
            let result = encode(&absolute_schema, &values, &mut buf, EncodeOptions::default()).unwrap();
            buf[..result.len].to_vec()
        }).collect()
    };

    decode_group.bench_function("tspack_delta", |b| {
        let mut decoder = StreamDecoder::new();
        b.iter(|| {
            decoder.reset();
            for frame in &delta_frames {
                let _ = decoder.decode(black_box(&delta_schema), black_box(frame));
            }
        })
    });

    decode_group.bench_function("tspack_absolute", |b| {
        b.iter(|| {
            for frame in &absolute_frames {
                let _ = decode(black_box(&absolute_schema), black_box(frame));
            }
        })
    });

    decode_group.finish();
}

/// Solar weather sensor data from Kaggle
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SolarWeatherData {
    #[serde(rename = "DATE_TIME")]
    date_time: String,
    #[serde(rename = "PLANT_ID")]
    plant_id: u64,
    #[serde(rename = "SOURCE_KEY")]
    source_key: String,
    #[serde(rename = "AMBIENT_TEMPERATURE")]
    ambient_temp: f32,
    #[serde(rename = "MODULE_TEMPERATURE")]
    module_temp: f32,
    #[serde(rename = "IRRADIATION")]
    irradiation: f32,
}

/// Load solar weather data from CSV
fn load_solar_data() -> Vec<SolarWeatherData> {
    let path = Path::new("solor_power/Plant_1_Weather_Sensor_Data.csv");
    if !path.exists() {
        println!("Solar data not found at {:?}, skipping solar benchmark", path);
        return vec![];
    }

    let mut reader = csv::Reader::from_path(path).expect("Failed to open CSV");
    reader
        .deserialize()
        .filter_map(|r| r.ok())
        .collect()
}

/// Benchmark with real solar IoT data
fn bench_solar(c: &mut Criterion) {
    use tspack::{DeltaSchema, StreamEncoder, StreamDecoder};

    let solar_data = load_solar_data();
    if solar_data.is_empty() {
        println!("Skipping solar benchmark - no data");
        return;
    }

    println!("\n=== Solar Power Dataset ===");
    println!("Samples: {}", solar_data.len());

    // Calculate actual ranges from data
    let (min_amb, max_amb) = solar_data.iter()
        .fold((f32::MAX, f32::MIN), |(min, max), d| {
            (min.min(d.ambient_temp), max.max(d.ambient_temp))
        });
    let (min_mod, max_mod) = solar_data.iter()
        .fold((f32::MAX, f32::MIN), |(min, max), d| {
            (min.min(d.module_temp), max.max(d.module_temp))
        });
    let (min_irr, max_irr) = solar_data.iter()
        .fold((f32::MAX, f32::MIN), |(min, max), d| {
            (min.min(d.irradiation), max.max(d.irradiation))
        });

    println!("Ambient Temp: {:.1} ~ {:.1} C", min_amb, max_amb);
    println!("Module Temp:  {:.1} ~ {:.1} C", min_mod, max_mod);
    println!("Irradiation:  {:.3} ~ {:.3}", min_irr, max_irr);

    // Calculate max deltas (15-minute intervals)
    let max_deltas: Vec<(f32, f32, f32)> = solar_data.windows(2)
        .map(|w| (
            (w[1].ambient_temp - w[0].ambient_temp).abs(),
            (w[1].module_temp - w[0].module_temp).abs(),
            (w[1].irradiation - w[0].irradiation).abs(),
        ))
        .collect();

    let max_delta_amb = max_deltas.iter().map(|d| d.0).fold(0.0f32, f32::max);
    let max_delta_mod = max_deltas.iter().map(|d| d.1).fold(0.0f32, f32::max);
    let max_delta_irr = max_deltas.iter().map(|d| d.2).fold(0.0f32, f32::max);

    println!("Max Delta Ambient: {:.2} C", max_delta_amb);
    println!("Max Delta Module:  {:.2} C", max_delta_mod);
    println!("Max Delta Irradiation: {:.4}", max_delta_irr);

    // Define schemas with some margin
    let margin = 1.1;
    let delta_schema = DeltaSchema::new(
        1,
        &[
            (Field::new(min_amb - 5.0, max_amb + 5.0, 0.1), max_delta_amb * margin),
            (Field::new(min_mod - 5.0, max_mod + 5.0, 0.1), max_delta_mod * margin),
            (Field::new(0.0, max_irr + 0.1, 0.001), max_delta_irr * margin),
        ],
        32,
    );

    let absolute_schema = Schema::new(
        1,
        &[
            Field::new(min_amb - 5.0, max_amb + 5.0, 0.1),
            Field::new(min_mod - 5.0, max_mod + 5.0, 0.1),
            Field::new(0.0, max_irr + 0.1, 0.001),
        ],
    );

    // Compression comparison
    {
        let mut encoder = StreamEncoder::new();
        let mut buf = [0u8; 32];
        let mut total_delta_bytes = 0usize;
        let mut total_absolute_bytes = 0usize;
        let mut absolute_frames = 0usize;
        let mut delta_frames = 0usize;

        for data in &solar_data {
            let values = [
                Some(data.ambient_temp),
                Some(data.module_temp),
                Some(data.irradiation),
            ];

            // Delta encoding
            let (len, is_abs) = encoder.encode(&delta_schema, &values, &mut buf).unwrap();
            total_delta_bytes += len;
            if is_abs { absolute_frames += 1; } else { delta_frames += 1; }

            // Absolute encoding
            buf.fill(0);
            let result = encode(&absolute_schema, &values, &mut buf, EncodeOptions::default()).unwrap();
            total_absolute_bytes += result.len;
        }

        let json_bytes: usize = solar_data.iter()
            .map(|d| serde_json::to_vec(d).unwrap().len())
            .sum();

        let msgpack_bytes: usize = solar_data.iter()
            .map(|d| rmp_serde::to_vec(d).unwrap().len())
            .sum();

        println!("\n=== Solar Data Compression ({} samples) ===", solar_data.len());
        println!("tspack delta:    {:>7} bytes ({} abs + {} delta)",
                 total_delta_bytes, absolute_frames, delta_frames);
        println!("tspack absolute: {:>7} bytes", total_absolute_bytes);
        println!("MessagePack:     {:>7} bytes", msgpack_bytes);
        println!("JSON:            {:>7} bytes", json_bytes);
        println!("Delta savings:   {:.1}% vs absolute",
                 (1.0 - total_delta_bytes as f64 / total_absolute_bytes as f64) * 100.0);
        println!("Delta vs MsgPack: {:.1}x smaller",
                 msgpack_bytes as f64 / total_delta_bytes as f64);
        println!("Delta vs JSON:    {:.1}x smaller",
                 json_bytes as f64 / total_delta_bytes as f64);
        println!("==============================================\n");
    }

    // Speed benchmarks
    let sample_count = solar_data.len().min(1000);
    let test_data: Vec<_> = solar_data.iter().take(sample_count).collect();

    let mut group = c.benchmark_group("solar_encode");
    group.throughput(Throughput::Elements(sample_count as u64));

    group.bench_function("tspack_delta", |b| {
        let mut encoder = StreamEncoder::new();
        let mut buf = [0u8; 32];
        b.iter(|| {
            encoder.reset();
            for data in &test_data {
                let values = [
                    Some(data.ambient_temp),
                    Some(data.module_temp),
                    Some(data.irradiation),
                ];
                let _ = encoder.encode(black_box(&delta_schema), black_box(&values), &mut buf);
            }
        })
    });

    group.bench_function("tspack_absolute", |b| {
        let mut buf = [0u8; 32];
        b.iter(|| {
            for data in &test_data {
                buf.fill(0);
                let values = [
                    Some(data.ambient_temp),
                    Some(data.module_temp),
                    Some(data.irradiation),
                ];
                let _ = encode(black_box(&absolute_schema), black_box(&values), &mut buf, EncodeOptions::default());
            }
        })
    });

    group.bench_function("json", |b| {
        b.iter(|| {
            for data in &test_data {
                let _ = serde_json::to_vec(black_box(*data));
            }
        })
    });

    group.bench_function("msgpack", |b| {
        b.iter(|| {
            for data in &test_data {
                let _ = rmp_serde::to_vec(black_box(*data));
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_size, bench_batch, bench_delta, bench_solar);
criterion_main!(benches);
