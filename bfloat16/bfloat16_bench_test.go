package bfloat16_test

import (
	"math"
	"testing"

	"github.com/gocnn/half/bfloat16"
)

func BenchmarkFrombits(b *testing.B) {
	bits := uint16(0x3f80)
	for b.Loop() {
		bfloat16.Frombits(bits)
	}
}

func BenchmarkFloat32(b *testing.B) {
	f := bfloat16.Fromfloat32(math.Pi)
	for b.Loop() {
		f.Float32()
	}
}

func BenchmarkFromfloat32(b *testing.B) {
	for b.Loop() {
		bfloat16.Fromfloat32(math.Pi)
	}
}

func BenchmarkFromfloat32Inf(b *testing.B) {
	inf := float32(math.Inf(1))
	for b.Loop() {
		bfloat16.Fromfloat32(inf)
	}
}

func BenchmarkFromfloat32NaN(b *testing.B) {
	nan := float32(math.NaN())
	for b.Loop() {
		bfloat16.Fromfloat32(nan)
	}
}

func BenchmarkIsNaN(b *testing.B) {
	f := bfloat16.NaN()
	for b.Loop() {
		f.IsNaN()
	}
}

func BenchmarkIsInf(b *testing.B) {
	f := bfloat16.Inf(1)
	for b.Loop() {
		f.IsInf(0)
	}
}

func BenchmarkString(b *testing.B) {
	f := bfloat16.Fromfloat32(math.Pi)
	for b.Loop() {
		_ = f.String()
	}
}
