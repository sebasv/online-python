#!/bin/bash
version_file="src/version.rs";

# bump the build number
IFS="." read -ra  version <<< $(tail src/version.rs | grep "pub const PY_VERSION:" | grep -o "[0-9]*\.[0-9]*\.[0-9]*");
major=${version[0]};
minor=${version[1]};
build=${version[2]};
echo "pub const PY_VERSION: &str = \"$major.$minor.$(($build+1))\";" > $version_file;

# do a build
cargo build --release
rc=$?; 
if [[ $rc != 0 ]]; 
then 
echo 'build failed'; 

# unbump the build number
echo "const PY_VERSION: &str = \"$major.$minor.$build\";" > $version_file;

else
cp target/release/libonline_python.so online_python.so;
cp target/release/libonline_python.so /home/sebas/stack/Code/backtest/online_python.so;
python -i -c"import online_python as op;print(op.__version__);print(dir(op))";
fi 
