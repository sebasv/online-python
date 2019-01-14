cargo build --release
rc=$?; 
if [[ $rc != 0 ]]; 
then echo 'build failed'; 
else
cp target/release/libonline_python.so online_python.so;
cp target/release/libonline_python.so /home/sebas/stack/Code/backtest/online_python.so;
python -i -c"import online_python as op;print(op.__version__);print(dir(op))";
fi # break if build failed
