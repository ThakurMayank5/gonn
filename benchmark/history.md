- v0.1.3

```bash
goos: windows
goarch: amd64
pkg: github.com/ThakurMayank5/gonn/benchmark
cpu: 13th Gen Intel(R) Core(TM) i7-13650HX
BenchmarkDenseForward-20           23841             50995 ns/op      1696 B/op           4 allocs/op
BenchmarkBackpropagation-20          147          12056676 ns/op   1141776 B/op         794 allocs/op
BenchmarkTrainingStep-20              96          12210747 ns/op   1141777 B/op         794 allocs/op
PASS
ok      github.com/ThakurMayank5/gonn/benchmark 6.232s
```

- v0.1.4 - Inference Optimized Heavily

```bash
goos: windows
goarch: amd64
pkg: github.com/ThakurMayank5/gonn/benchmark
cpu: 13th Gen Intel(R) Core(TM) i7-13650HX
BenchmarkDenseForward
BenchmarkDenseForward-20          108864             32663 ns/op               0 B/op          0 allocs/op
BenchmarkDenseForward-20          112492             31659 ns/op               0 B/op          0 allocs/op
BenchmarkDenseForward-20          113670             32303 ns/op               0 B/op          0 allocs/op
PASS
ok      github.com/ThakurMayank5/gonn/benchmark 12.501s
```
