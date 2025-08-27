# FixedAI: Leiden Community Detection

[![Crates.io](https://img.shields.io/crates/v/fa-leiden-cd.svg)](https://crates.io/crates/fa-leiden-cd)
[![Documentation](https://docs.rs/fa-leiden-cd/badge.svg)](https://docs.rs/fa-leiden-cd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/fixed-ai/fa-leiden-cd/blob/main/LICENSE)
[![CI](https://github.com/fixed-ai/fa-leiden-cd/workflows/CI/badge.svg)](https://github.com/fixed-ai/fa-leiden-cd/actions)
[![codecov](https://codecov.io/gh/fixed-ai/fa-leiden-cd/graph/badge.svg?token=Mu3ifjvEeW)](https://codecov.io/gh/fixed-ai/fa-leiden-cd)

`fa-leiden-cd` is a Rust implementation of the Leiden algorithm for community detection in large networks.

## Installation

```bash
cargo add fa-leiden-cd
```

## Use Cases

1. Agent/RAG system: Leiden algorithm is the SOTA method in community detection, and eases the build of a graph-based representation of knowledge databases.
2. Social network analysis: Identify communities within social networks to understand user behavior and interactions.
3. Applications: Unlike Python/Java, a Rust-based library can be easily integrated into customer-facing applications for all platforms, including web, desktop, mobile and embedded systems.

## Usage

```rust
use fa_leiden_cd::{Graph, TrivialModularityOptimizer};

fn test() {
    let mut nodes: HashMap<&'static str, usize> = HashMap::new();
    let mut g = Graph::new();

    let edges: &[(&'static str, &'static str, f32)] = &[
        ("Fortran", "C", 0.5),
        ("Fortran", "LISP", 0.3),
        ("Fortran", "MATLAB", 0.6),
        ("C", "C++", 0.9),
        ("C", "Go", 0.6),
        ("LISP", "ML", 0.5),
        ("LISP", "OCaml", 0.2),
        ("LISP", "Haskell", 0.2),
        ("LISP", "Ruby", 0.5),
        ("LISP", "Julia", 0.6),
        ("ML", "OCaml", 0.8),
        ("ML", "Haskell", 0.5),
        ("OCaml", "Haskell", 0.3),
        ("OCaml", "F#", 0.6),
        ("Haskell", "Julia", 0.2),
        ("C++", "Python", 0.32),
        ("C++", "Ruby", 0.2),
        ("C++", "C#", 0.5),
        ("Python", "F#", 0.2),
        ("Python", "Julia", 0.4),
        ("C#", "F#", 0.3),
    ];

    for (from, to, weight) in edges.iter() {
        let from_id = *nodes.entry(from).or_insert_with(|| g.add_node(from));
        let to_id = *nodes.entry(to).or_insert_with(|| g.add_node(to));
        g.add_edge(from_id, to_id, (), *weight);
    }

    let mut optimizer = TrivialModularityOptimizer {
            parallel_scale: 128,
            tol: 1e-11,
        };

        let hierarchy = g.leiden(Some(100), &mut optimizer);
        for (i, node) in hierarchy.node_data_slice().iter().enumerate() {
            println!("community {}:", i);
            node.collect_nodes(&|i| {
                let n = g.node_data_slice()[i];
                println!("     {}", n);
            });
        }
}

// community 0:
//      Haskell
//      ML
//      F#
//      OCaml
// community 1:
//      Python
//      Julia
// community 2:
//      Fortran
//      MATLAB
//      Go
//      C
//      C#
//      C++
// community 3:
//      Ruby
//      LISP
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.