use std::path::Path;

use polyfit::{
    basis_select,
    score::{Aic, Bic},
    statistics::DegreeBound,
};

fn main() {
    //
    // First argument is the path to a CSV or JSON file with (x, y) data points.
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("Please provide a path to a CSV or JSON file with (x, y) data points.");
        std::process::exit(1);
    };
    let path = Path::new(&path);

    let mut degree_bound = DegreeBound::Relaxed;
    let mut method = "aic".to_string();
    for arg in std::env::args().skip(2) {
        if let Some(option) = arg.strip_prefix("degree_bound=") {
            match option {
                "relaxed" => degree_bound = DegreeBound::Relaxed,
                "conservative" => degree_bound = DegreeBound::Conservative,
                _ => match str::parse::<usize>(option) {
                    Ok(value) => degree_bound = DegreeBound::Custom(value),
                    Err(_) => {
                        eprintln!("Invalid degree_bound value: {}", option);
                        std::process::exit(1);
                    }
                },
            }
        }

        if let Some(option) = arg.strip_prefix("method=") {
            method = option.to_string();
        }

        if arg == "help" || arg == "--help" || arg == "-h" {
            eprintln!("Usage: basis_select <path> [degree_bound=relaxed|conservative|<number>] [method=aic|bic]");
            std::process::exit(0);
        }
    }

    let Ok(contents) = std::fs::read_to_string(path) else {
        eprintln!("Failed to read file: {}", path.display());
        std::process::exit(1);
    };

    let data: Vec<(f64, f64)> = match path.extension().and_then(|s| s.to_str()) {
        Some("csv") => {
            // Simple CSV parser: expects two columns, x and y, with a header row.
            let mut lines = contents.lines();
            let mut data = Vec::new();

            fn parse_line(line: &str) -> Option<(f64, f64)> {
                let mut parts = line.split(',').map(str::trim);
                let x = parts.next()?.parse().ok()?;
                let y = parts.next()?.parse().ok()?;
                Some((x, y))
            }

            //
            // First line, if we get a parsing error, we assume it's a header and skip it.
            if let Some(first_line) = lines.next() {
                if parse_line(first_line).is_some() {
                    if let Some(point) = parse_line(first_line) {
                        data.push(point);
                    }
                }
            }

            //
            // The rest we are strict.
            for (i, line) in lines.enumerate() {
                match parse_line(line) {
                    Some(point) => data.push(point),
                    None => {
                        eprintln!("Failed to parse line {}: {}", i + 2, line);
                        std::process::exit(1);
                    }
                }
            }

            data
        }
        Some("json") => serde_json::from_str(&contents).unwrap_or_else(|err| {
            eprintln!("Failed to parse JSON: {}", err);
            std::process::exit(1);
        }),

        _ => {
            eprintln!("Unsupported file format: {}", path.display());
            std::process::exit(1);
        }
    };

    match method.as_str() {
        "aic" => basis_select!(&data, degree_bound, &Aic),
        "bic" => basis_select!(&data, degree_bound, &Bic),
        _ => {
            eprintln!("Unsupported method: {}", method);
            std::process::exit(1);
        }
    }

    std::process::exit(0);
}
