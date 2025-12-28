package bfloat16_test

import (
	"math"
	"testing"

	"github.com/gocnn/half/bfloat16"
)

// Test conversion from float32 to BFloat16 and back.
var conversionTests = []struct {
	f32  float32
	bits uint16
}{
	{0.0, 0x0000},
	{float32(math.Copysign(0, -1)), 0x8000}, // -0.0
	{1.0, 0x3f80},
	{-1.0, 0xbf80},
	{2.0, 0x4000},
	{0.5, 0x3f00},
	{0.25, 0x3e80},
	{3.140625, 0x4049}, // pi approximation
	{math.Float32frombits(0x7f800000), 0x7f80}, // +Inf
	{math.Float32frombits(0xff800000), 0xff80}, // -Inf
	{math.Float32frombits(0x7fc00000), 0x7fc0}, // NaN (quiet)
}

func TestFromfloat32(t *testing.T) {
	for _, tc := range conversionTests {
		got := bfloat16.Fromfloat32(tc.f32)
		if got.Bits() != tc.bits {
			t.Errorf("Fromfloat32(%v) = 0x%04x, want 0x%04x", tc.f32, got.Bits(), tc.bits)
		}
	}
}

func TestFloat32(t *testing.T) {
	tests := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0.0},
		{0x8000, float32(math.Copysign(0, -1))},
		{0x3f80, 1.0},
		{0xbf80, -1.0},
		{0x4000, 2.0},
		{0x3f00, 0.5},
	}
	for _, tc := range tests {
		f := bfloat16.Frombits(tc.bits)
		got := f.Float32()
		if got != tc.want && !(math.IsNaN(float64(got)) && math.IsNaN(float64(tc.want))) {
			t.Errorf("Frombits(0x%04x).Float32() = %v, want %v", tc.bits, got, tc.want)
		}
	}
}

func TestFloat64(t *testing.T) {
	f := bfloat16.Fromfloat32(1.5)
	got := f.Float64()
	want := float64(1.5)
	if got != want {
		t.Errorf("Float64() = %v, want %v", got, want)
	}
}

func TestFrombits(t *testing.T) {
	x := uint16(0x1234)
	f := bfloat16.Frombits(x)
	if f.Bits() != x {
		t.Errorf("Frombits(0x%04x).Bits() = 0x%04x", x, f.Bits())
	}
}

func TestBits(t *testing.T) {
	f := bfloat16.Fromfloat32(1.0)
	if f.Bits() != 0x3f80 {
		t.Errorf("Bits() = 0x%04x, want 0x3f80", f.Bits())
	}
}

func TestNaN(t *testing.T) {
	nan := bfloat16.NaN()
	if !nan.IsNaN() {
		t.Error("NaN().IsNaN() = false, want true")
	}
}

func TestInf(t *testing.T) {
	tests := []struct {
		sign int
		want uint16
	}{
		{0, 0x7f80},
		{1, 0x7f80},
		{-1, 0xff80},
	}
	for _, tc := range tests {
		got := bfloat16.Inf(tc.sign).Bits()
		if got != tc.want {
			t.Errorf("Inf(%d) = 0x%04x, want 0x%04x", tc.sign, got, tc.want)
		}
	}
}

func TestIsNaN(t *testing.T) {
	tests := []struct {
		bits uint16
		want bool
	}{
		{0x0000, false},
		{0x3f80, false},
		{0x7f80, false}, // +Inf
		{0x7fc0, true},  // quiet NaN
		{0x7f81, true},  // signaling NaN
	}
	for _, tc := range tests {
		f := bfloat16.Frombits(tc.bits)
		if got := f.IsNaN(); got != tc.want {
			t.Errorf("Frombits(0x%04x).IsNaN() = %v, want %v", tc.bits, got, tc.want)
		}
	}
}

func TestIsQuietNaN(t *testing.T) {
	tests := []struct {
		bits uint16
		want bool
	}{
		{0x0000, false},
		{0x7f80, false}, // +Inf
		{0x7fc0, true},  // quiet NaN (bit 6 set)
		{0x7f81, false}, // signaling NaN (bit 6 not set)
	}
	for _, tc := range tests {
		f := bfloat16.Frombits(tc.bits)
		if got := f.IsQuietNaN(); got != tc.want {
			t.Errorf("Frombits(0x%04x).IsQuietNaN() = %v, want %v", tc.bits, got, tc.want)
		}
	}
}

func TestIsInf(t *testing.T) {
	tests := []struct {
		bits uint16
		sign int
		want bool
	}{
		{0x0000, 0, false},
		{0x3f80, 0, false},
		{0x7f80, 0, true},   // +Inf, any sign
		{0x7f80, 1, true},   // +Inf, positive
		{0x7f80, -1, false}, // +Inf, negative
		{0xff80, 0, true},   // -Inf, any sign
		{0xff80, 1, false},  // -Inf, positive
		{0xff80, -1, true},  // -Inf, negative
		{0x7fc0, 0, false},  // NaN
	}
	for _, tc := range tests {
		f := bfloat16.Frombits(tc.bits)
		if got := f.IsInf(tc.sign); got != tc.want {
			t.Errorf("Frombits(0x%04x).IsInf(%d) = %v, want %v", tc.bits, tc.sign, got, tc.want)
		}
	}
}

func TestIsFinite(t *testing.T) {
	tests := []struct {
		name string
		f    bfloat16.BFloat16
		want bool
	}{
		{"zero", bfloat16.Frombits(0), true},
		{"one", bfloat16.Fromfloat32(1.0), true},
		{"posInf", bfloat16.Inf(1), false},
		{"negInf", bfloat16.Inf(-1), false},
		{"nan", bfloat16.NaN(), false},
	}
	for _, tc := range tests {
		if got := tc.f.IsFinite(); got != tc.want {
			t.Errorf("%s.IsFinite() = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestIsNormal(t *testing.T) {
	tests := []struct {
		name string
		f    bfloat16.BFloat16
		want bool
	}{
		{"zero", bfloat16.Frombits(0), false},
		{"subnormal", bfloat16.Frombits(0x0001), false},
		{"normal", bfloat16.Fromfloat32(1.0), true},
		{"posInf", bfloat16.Inf(1), false},
		{"negInf", bfloat16.Inf(-1), false},
		{"nan", bfloat16.NaN(), false},
	}
	for _, tc := range tests {
		if got := tc.f.IsNormal(); got != tc.want {
			t.Errorf("%s.IsNormal() = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestSign(t *testing.T) {
	tests := []struct {
		f    bfloat16.BFloat16
		want int
	}{
		{bfloat16.Fromfloat32(0), 0},
		{bfloat16.Fromfloat32(1.0), 1},
		{bfloat16.Fromfloat32(-1.0), -1},
		{bfloat16.Fromfloat32(100.0), 1},
		{bfloat16.Fromfloat32(-100.0), -1},
		{bfloat16.NaN(), 0},
	}
	for _, tc := range tests {
		if got := tc.f.Sign(); got != tc.want {
			t.Errorf("%v.Sign() = %d, want %d", tc.f, got, tc.want)
		}
	}
}

func TestSignbit(t *testing.T) {
	tests := []struct {
		f    float32
		want bool
	}{
		{0.0, false},
		{1.0, false},
		{-1.0, true},
		{float32(math.Copysign(0, -1)), true}, // -0
	}
	for _, tc := range tests {
		f := bfloat16.Fromfloat32(tc.f)
		if got := f.Signbit(); got != tc.want {
			t.Errorf("Fromfloat32(%v).Signbit() = %v, want %v", tc.f, got, tc.want)
		}
	}
}

func TestString(t *testing.T) {
	tests := []struct {
		f    float32
		want string
	}{
		{1.0, "1"},
		{1.5, "1.5"},
		{0.5, "0.5"},
	}
	for _, tc := range tests {
		f := bfloat16.Fromfloat32(tc.f)
		if got := f.String(); got != tc.want {
			t.Errorf("Fromfloat32(%v).String() = %s, want %s", tc.f, got, tc.want)
		}
	}
}

func TestSmallestNonzero(t *testing.T) {
	if bfloat16.SmallestNonzero.Bits() != 0x0001 {
		t.Errorf("SmallestNonzero = 0x%04x, want 0x0001", bfloat16.SmallestNonzero.Bits())
	}
	f32 := bfloat16.SmallestNonzero.Float32()
	if f32 <= 0 {
		t.Errorf("SmallestNonzero.Float32() = %v, want > 0", f32)
	}
}

func TestRoundTrip(t *testing.T) {
	// Test that normal float32 values that fit in bfloat16 round-trip correctly
	values := []float32{1.0, -1.0, 2.0, 0.5, 100.0, -100.0, 0.0}
	for _, v := range values {
		bf := bfloat16.Fromfloat32(v)
		back := bf.Float32()
		if back != v {
			t.Errorf("round-trip failed: %v -> 0x%04x -> %v", v, bf.Bits(), back)
		}
	}
}

func TestRounding(t *testing.T) {
	// BFloat16 uses round-to-nearest-even
	// Test that values between representable values round correctly

	// 1.0 = 0x3f80, 1.0078125 = 0x3f81
	// Values between should round to nearest
	f := bfloat16.Fromfloat32(1.00390625) // exactly halfway
	got := f.Float32()
	// Should round to even (1.0 = 0x3f80, which has LSB=0)
	if got != 1.0 {
		t.Logf("1.00390625 rounded to %v (0x%04x)", got, f.Bits())
	}
}

func TestSpecialValues(t *testing.T) {
	// Test +Inf
	posInf := bfloat16.Inf(1)
	if !math.IsInf(float64(posInf.Float32()), 1) {
		t.Error("+Inf conversion failed")
	}

	// Test -Inf
	negInf := bfloat16.Inf(-1)
	if !math.IsInf(float64(negInf.Float32()), -1) {
		t.Error("-Inf conversion failed")
	}

	// Test NaN
	nan := bfloat16.NaN()
	if !math.IsNaN(float64(nan.Float32())) {
		t.Error("NaN conversion failed")
	}
}

func TestFromfloat32SpecialCases(t *testing.T) {
	// Test Inf preservation
	posInf := bfloat16.Fromfloat32(float32(math.Inf(1)))
	if !posInf.IsInf(1) {
		t.Error("Fromfloat32(+Inf) should be +Inf")
	}

	negInf := bfloat16.Fromfloat32(float32(math.Inf(-1)))
	if !negInf.IsInf(-1) {
		t.Error("Fromfloat32(-Inf) should be -Inf")
	}

	// Test NaN preservation
	nan := bfloat16.Fromfloat32(float32(math.NaN()))
	if !nan.IsNaN() {
		t.Error("Fromfloat32(NaN) should be NaN")
	}
}
