package float16

import (
	"errors"
	"math"
	"strconv"
)

// Precision indicates whether the conversion to Float16 is
// exact, subnormal without dropped bits, inexact, underflow, or overflow.
type Precision int

const (
	PrecisionExact     Precision = iota // exact conversion, round-trips
	PrecisionUnknown                    // subnormal, may or may not round-trip
	PrecisionInexact                    // dropped bits, cannot round-trip
	PrecisionUnderflow                  // underflow, cannot round-trip
	PrecisionOverflow                   // overflow, cannot round-trip
)

// Float16 represents IEEE 754 half-precision floating-point numbers (binary16).
type Float16 uint16

// Float16 bit masks and special values.
const (
	signMask Float16 = 0x8000
	expMask  Float16 = 0x7c00
	coefMask Float16 = 0x03ff
	qnanBit  Float16 = 0x0200

	posInf Float16 = 0x7c00
	negInf Float16 = 0xfc00
	qNaN   Float16 = 0x7e01
	sNaN   Float16 = 0x7c01
)

// Float32 bit layout constants.
const (
	f32Sign     uint32 = 0x80000000
	f32Exp      uint32 = 0x7f800000
	f32Coef     uint32 = 0x007fffff
	f32Shift    uint32 = 23
	f32Bias     int32  = 127
	f16Bias     int32  = 15
	f16ExpMax   uint32 = 0x1f
	f32Hidden   uint32 = 0x00800000
	f32RoundBit uint32 = 0x00001000
)

// SmallestNonzero is the smallest positive nonzero Float16 (≈5.96e-08).
const SmallestNonzero = Float16(0x0001)

// ErrInvalidNaN indicates the input was not a valid NaN.
var ErrInvalidNaN = errors.New("float16: expected NaN input")

// PrecisionFromfloat32 returns Precision without performing
// the conversion.  Conversions from both Infinity and NaN
// values will always report PrecisionExact even if NaN payload
// or NaN-Quiet-Bit is lost. This function is kept simple to
// allow inlining and run < 0.5 ns/op, to serve as a fast filter.
func PrecisionFromfloat32(f32 float32) Precision {
	u32 := math.Float32bits(f32)

	if u32 == 0 || u32 == f32Sign {
		return PrecisionExact
	}

	const dropMask = f32Coef >> 10

	exp := int32((u32&f32Exp)>>f32Shift) - f32Bias
	coef := u32 & f32Coef

	if exp == 128 {
		return PrecisionExact
	}

	// https://en.wikipedia.org/wiki/Half-precision_floating-point_format says,
	// "Decimals between 2^−24 (minimum positive subnormal) and 2^−14 (maximum subnormal): fixed interval 2^−24"
	if exp < -24 {
		return PrecisionUnderflow
	}
	if exp > 15 {
		return PrecisionOverflow
	}
	if coef&dropMask != 0 {
		return PrecisionInexact
	}

	if exp < -14 {
		// Subnormals. Caller may want to test these further.
		// There are 2046 subnormals that can successfully round-trip f32->f16->f32
		// and 20 of those 2046 have 32-bit input coef == 0.
		// RFC 7049 and 7049bis Draft 12 don't precisely define "preserves value"
		// so some protocols and libraries will choose to handle subnormals differently
		// when deciding to encode them to CBOR float32 vs float16.
		return PrecisionUnknown
	}

	return PrecisionExact
}

// Frombits returns the float16 number corresponding to the IEEE 754 binary16
// representation u16, with the sign bit of u16 and the result in the same bit
// position. Frombits(Bits(x)) == x.
func Frombits(u16 uint16) Float16 {
	return Float16(u16)
}

// Fromfloat32 returns a Float16 value converted from f32. Conversion uses
// IEEE default rounding (nearest int, with ties to even).
func Fromfloat32(f32 float32) Float16 {
	return Float16(f32bitsToF16bits(math.Float32bits(f32)))
}

// FromNaN32ps converts a float32 NaN to Float16 NaN preserving sign and payload.
// Returns ErrInvalidNaN if input is not NaN.
func FromNaN32ps(nan float32) (Float16, error) {
	u32 := math.Float32bits(nan)
	sign := u32 & f32Sign
	exp := u32 & f32Exp
	coef := u32 & f32Coef

	if exp != f32Exp || coef == 0 {
		return sNaN, ErrInvalidNaN
	}

	f16 := Float16((sign >> 16) | uint32(expMask) | (coef >> 13))
	if f16&coefMask == 0 {
		f16 |= 0x0001
	}
	return f16, nil
}

// NaN returns a Float16 quiet NaN (0x7e01).
func NaN() Float16 { return qNaN }

// Inf returns positive infinity if sign >= 0, negative infinity otherwise.
func Inf(sign int) Float16 {
	if sign >= 0 {
		return posInf
	}
	return negInf
}

// Float32 converts f to float32 (lossless).
func (f Float16) Float32() float32 {
	return math.Float32frombits(f16bitsToF32bits(uint16(f)))
}

// Bits returns the IEEE 754 binary16 representation.
func (f Float16) Bits() uint16 { return uint16(f) }

// IsNaN reports whether f is NaN.
func (f Float16) IsNaN() bool {
	return f&expMask == expMask && f&coefMask != 0
}

// IsQuietNaN reports whether f is a quiet NaN.
func (f Float16) IsQuietNaN() bool {
	return f&expMask == expMask && f&coefMask != 0 && f&qnanBit != 0
}

// IsInf reports whether f is infinity. sign > 0: +Inf, sign < 0: -Inf, sign == 0: either.
func (f Float16) IsInf(sign int) bool {
	if sign == 0 {
		return f == posInf || f == negInf
	}
	if sign > 0 {
		return f == posInf
	}
	return f == negInf
}

// IsFinite reports whether f is finite (not Inf or NaN).
func (f Float16) IsFinite() bool { return f&expMask != expMask }

// IsNormal reports whether f is a normal number (not zero, subnormal, Inf, or NaN).
func (f Float16) IsNormal() bool {
	e := f & expMask
	return e != 0 && e != expMask
}

// Signbit reports whether f is negative or negative zero.
func (f Float16) Signbit() bool { return f&signMask != 0 }

// String implements fmt.Stringer.
func (f Float16) String() string {
	return strconv.FormatFloat(float64(f.Float32()), 'f', -1, 32)
}

func f16bitsToF32bits(b uint16) uint32 {
	s := uint32(b>>15) << 31
	e := uint32((b >> 10) & 0x1f)
	m := uint32(b&0x03ff) << 13

	if e == f16ExpMax {
		if m == 0 {
			return s | f32Exp
		}
		return s | f32Exp | (1 << 22) | m
	}

	if e == 0 {
		if m == 0 {
			return s
		}
		e = 1
		for m&f32Exp == 0 {
			m <<= 1
			e--
		}
		m &= f32Coef
	}

	return s | (uint32(int32(e)+f32Bias-f16Bias) << 23) | m
}

func f32bitsToF16bits(b uint32) uint16 {
	s := b & f32Sign
	e := b & f32Exp
	m := b & f32Coef

	if e == f32Exp {
		q := uint32(0)
		if m != 0 {
			q = uint32(qnanBit)
		}
		return uint16((s >> 16) | uint32(expMask) | q | (m >> 13))
	}

	hs := s >> 16
	ue := int32(e>>f32Shift) - f32Bias
	he := ue + f16Bias

	if he >= int32(f16ExpMax) {
		return uint16(hs | uint32(expMask))
	}

	if he <= 0 {
		if 14-he > 24 {
			return uint16(hs)
		}
		c := m | f32Hidden
		hm := c >> uint32(14-he)
		rb := uint32(1) << uint32(13-he)
		if c&rb != 0 && c&(3*rb-1) != 0 {
			hm++
		}
		return uint16(hs | hm)
	}

	hex := uint32(he) << 10
	hm := m >> 13
	if m&f32RoundBit != 0 && m&(3*f32RoundBit-1) != 0 {
		return uint16((hs | hex | hm) + 1)
	}
	return uint16(hs | hex | hm)
}
