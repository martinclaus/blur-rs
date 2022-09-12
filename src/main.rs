//! Experimental architecture for kernel based computations on 2D arrays.

mod data;
mod executor;
mod kernel;

use crate::executor::{
    RayonIteratorExecutor, RayonJoinExecutor, RayonScopeExecutor, SerialExecutor,
    ThreadChannelExecutor, ThreadPoolExecutor, ThreadSharedMutableStateExecutor,
};

fn main() {
    let (fw1, fw3) = (35, 5);

    // serial execution
    println!(
        "{:<fw1$} {:>3} {:>fw3$}",
        "SerialExecutor",
        1,
        run_benchmark(SerialExecutor {}).as_millis()
    );

    // parallel execution
    for n_threads in [4, 8, 16] {
        {
            println!(
                "{:<fw1$} {:>3} {:>fw3$}",
                "RayonIteratorExecutor",
                n_threads,
                make_thread_pool(n_threads)
                    .install(|| run_benchmark(RayonIteratorExecutor {}))
                    .as_millis()
            );
        }
        {
            println!(
                "{:<fw1$} {:>3} {:>fw3$}",
                "RayonScopeExecutor",
                n_threads,
                make_thread_pool(n_threads)
                    .install(|| run_benchmark(RayonScopeExecutor {}))
                    .as_millis()
            );
        }
        {
            println!(
                "{:<fw1$} {:>3} {:>fw3$}",
                "RayonJoinExecutor",
                n_threads,
                make_thread_pool(n_threads)
                    .install(|| run_benchmark(RayonJoinExecutor {}))
                    .as_millis()
            );
        }
        {
            println!(
                "{:<fw1$} {:>3} {:>fw3$}",
                "ThreadSharedMutableStateExecutor",
                n_threads,
                run_benchmark(ThreadSharedMutableStateExecutor { n_threads }).as_millis()
            );
        }

        {
            println!(
                "{:<fw1$} {:>3} {:>fw3$}",
                "ThreadChannelExecutor",
                n_threads,
                run_benchmark(ThreadChannelExecutor { n_threads }).as_millis()
            );
        }

        {
            println!(
                "{:<fw1$} {:>3} {:>fw3$}",
                "ThreadPoolExecutor",
                n_threads,
                run_benchmark(ThreadPoolExecutor::new(n_threads)).as_millis()
            );
        }
    }
}

use crate::{
    data::{Arr2D, Shape2D},
    executor::Executor,
    kernel::Blur,
};
use rayon::ThreadPool;
use std::time::{Duration, Instant};

/// Set up a Rayon thread pool for a given number of threads.
pub fn make_thread_pool(nthreads: usize) -> ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .unwrap()
}

/// Print slice of the final result
pub fn print_arr_sample(arr: Arr2D) {
    let i = arr.shape().0 / 2 - arr.shape().1 / 10 - 200;
    print!("Output Sample: ");
    (i..i + 6).for_each(|i| print! {"{:#.2e}, ", arr[[arr.shape().0 / 2, i]]});
    println!();
}

/// Mockup data and run a kernel on an executor
pub fn run_benchmark<E: Executor>(exec: E) -> Duration {
    // println!("{}", type_name::<E>());
    let shape = Shape2D(1000, 1000);
    let rep = 100;

    let mut d_in = Arr2D::full(0f64, shape);
    d_in.shape().iter().for_each(|ind| {
        if ((ind[0] as i64) - (shape.0 as i64) / 2).abs() < (shape.1 / 10) as i64
            && ((ind[1] as i64) - (shape.1 as i64) / 2).abs() < (shape.1 / 10) as i64
        {
            d_in[ind] = 1f64;
        } else {
            d_in[ind] = 0f64;
        }
    });

    let mut d_out = Arr2D::full(5f64, shape);

    let now = Instant::now();
    for _ in 0..rep {
        exec.run::<Blur>(&d_in, &mut d_out);
        exec.run::<Blur>(&d_out, &mut d_in);
    }
    // print_arr_sample(d_in);
    now.elapsed()
}
