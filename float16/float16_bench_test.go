package float16_test

import (
	"math"
	"testing"

	"github.com/gocnn/half/float16"
)

func BenchmarkFrombits(b *testing.B) {
	bits := uint16(float16.Fromfloat32(math.Pi))
	for b.Loop() {
		float16.Frombits(bits)
	}
}

func BenchmarkFloat32(b *testing.B) {
	f := float16.Fromfloat32(math.Pi)
	for b.Loop() {
		f.Float32()
	}
}

func BenchmarkFromfloat32Normal(b *testing.B) {
	for b.Loop() {
		float16.Fromfloat32(math.Pi)
	}
}

func BenchmarkFromfloat32NaN(b *testing.B) {
	nan := float32(math.NaN())
	for b.Loop() {
		float16.Fromfloat32(nan)
	}
}

func BenchmarkFromfloat32Subnormal(b *testing.B) {
	sub := math.Float32frombits(0x007fffff)
	for b.Loop() {
		float16.Fromfloat32(sub)
	}
}

func BenchmarkPrecisionFromfloat32(b *testing.B) {
	f := float32(0.00002)
	for b.Loop() {
		float16.PrecisionFromfloat32(f)
	}
}

func BenchmarkString(b *testing.B) {
	f := float16.Fromfloat32(math.Pi)
	for b.Loop() {
		_ = f.String()
	}
}
