# online-python

A Python module written in Rust that implements some loop-heavy routines needed in online portfolio optimization. 

## How to build

These are the latest instructions at the moment of writing. 
These might change as [`pyo3`](https://pyo3.rs/master/doc/pyo3/index.html) changes, if you run into issues check the pyo3 docs.

### Linux

Build using `cargo build --release` and copy `/target/release/online_python.so` to a place where Python can find it. 

### Windows

Build using `cargo build --release` and copy `/target/release/online_python.dll` to a place where Python can find it. 
Also, rename the file to `online_python.pyd`.

### MacOS
On Mac Os, you need to set additional linker arguments. 
One option is to compile with cargo rustc --release -- -C link-arg=-undefined -C link-arg=dynamic_lookup
, the other is to create a .cargo/config with the following content:

```
[target.x86_64-apple-darwin]
rustflags = [
  "-C", "link-arg=-undefined",
  "-C", "link-arg=dynamic_lookup",
]
```

Now copy `/target/release/online_python.dylib` to a place where Python can find it.
Also, rename the file to `online_python.so`.

## Usage in Python

Simply import using `import online_python`. 
Errors are not handled nicely, so if you supply erroneous arguments your Python instance will die.
The Python console will tell you what you did wrong, or where this code contains a bug.
