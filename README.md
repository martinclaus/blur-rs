# blur-rs

Experimental implementation of kernel based operations (e.g. gaussian blur, derivatives, ...) on arrays.
The implementation is experimental and should not be used anywhere! **Only limited testing is done!**.
The aim of this repo is to learn about the implementation of multi-threaded kernel based methods and their performance.

## API
The API is structured in the three main modules `data`, `kernel` and `executor` described below.

### data
`data` contains the basic types for dealing with a 2D array of floats. These types are:
- `Range2d`: a tuple struct of two ranges representing a index region in the 2D array. For this type, the `Iterator` and `From<Shape2D>` traits are implemented as well as a method to obtain a [rayon](https://docs.rs/rayon/latest/rayon/) parallel iterator. The iterators yield index arrays (`[uszie; 2]`) in row-major order, which is the memory alignment of the `Arr2D` type.
- `Shape2D`: a tuple struct representing the shape of an 2D array. A shape can be turned into an iterator over all valid index arrays and can be used to transform a index array (`[usize; 2]`) into a flatten index (`usize`) and vice versa.
- `Arr2D`: The main type of this module. It consists of an shape (`Shape2d`) and a data buffer on the heap. It is constructed via `Arr2D::full` and provides convenient access to it's shape, mutable and immutable iterators over the buffer and mutable and immutable slice views of the buffer. An `Arr2d` may be indexed via an index array, a flat index or a range. The memory layout is row-major.

### kernel
This module contains the `Kernel` trait which defines the interface of computational kernel types. 
A computational kernel represents an operation on a set of data that is applied identically for each possible location of the resulting dataset, although branching logic is permitted.
Note that a `Kernel` implementor only specifies the logic per location, not the logic how the computation is executed at all locations. This is the task of an `Executor` implementor (see next section).
Typically (but not necessarily), the operation uses input data neighboring the target location.
The result is bound to have the same shape as the data the kernel is operating on. 
A stereotype class of operations that can be represented by a kernel type are convolutions.
The `Kernel` trait requires to implement three associated functions:
- `shape`: returns the shape of a kernel. The shape determines how far 
- `map_index`: maps an index of a kernel element to an index array of the respective data element given the kernels' location of evaluation.
- `eval`: Evaluates the kernel for a given location using the provided data.

An example implementation is given with `Blur` for a `3x3` gaussian blur kernel.

### executor
The module defines the `Executor` trait which defines the interface for kernel executor types.
An executor evaluates a kernel operation at all locations of the resulting `Arr2D`.
By separating the per-location logic and grid iteration logic, it is possible to separate the concerns of evaluation (the kernel) and execution run-time (the executor).
The only required associated function is `run`, which executes the kernel evaluation for all locations.
There are several exemplary implementations:
- `SerialExecutor`: Simple serial (single-threaded) executor.
- `ThreadSharedMutableStateExecutor`: Multi-threaded executor based on [scoped threads of Rust's stdlib](https://doc.rust-lang.org/nightly/std/thread/fn.scope.html). The implementation uses unsafe Rust to avoid the use of `Arc` and `Mutex` for writing to memory shared by the threads.
- `ThreadChannelExecutor`: Also based on `std::thread::scope` but uses channels to communicate the results back to the main thread which mutates the global result buffer, hence no writing to shared memory. The trade-off are heap memory allocations for the result buffer of each thread. The number of allocations is minimized by handing back the thread result buffer to the threads for reuse. This logic is abstracted away in the `SPMDTask` type.
- `ThreadPoolExecutor`: Is also based on the `SPMDTask`, i.e. channel based communication and reuse of result buffers, but spawns a single pool of worker threads that are joined when the executor objects is dropped.
- `RayonExecutor`: Parallel executor build upon [Rayon's parallel iterator API](https://docs.rs/rayon/latest/rayon/iter/index.html).
- `RayonScopedExecutor`: Parallel executor build upon [Rayon's scoped threads](https://docs.rs/rayon/latest/rayon/fn.scope.html).
- `RayonJoinExecutor`: Parallel executor using [Rayon's Join API](https://docs.rs/rayon/latest/rayon/fn.join.html).