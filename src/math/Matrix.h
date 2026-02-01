#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <utility>
#include "Vector.h"

#ifdef __CUDACC__
#define GGG_CUDA __host__ __device__
#else
#define GGG_CUDA
#endif

namespace ggg
{
    namespace detail
    {
        template<typename T, std::size_t N>
        struct MatrixInverse;
    }

    template<typename T, std::size_t Rows, std::size_t Cols> requires (std::integral<T> || std::floating_point<T>)
    class Matrix
    {
    public:
        static_assert(Rows > 0);
        static_assert(Cols > 0);

        using value_type = T;
        static constexpr std::size_t rows = Rows;
        static constexpr std::size_t cols = Cols;

        GGG_CUDA constexpr Matrix()
            : m_columns([&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> std::array<Vector<T, rows>, cols>
                {
                    return [&] <std::size_t... Rs> (std::index_sequence<Rs...>) -> Vector<T, rows>
                        {
                            return { (Rs == Cs ? T{1} : T{0})... };
                        }(std::make_index_sequence<rows>());
                }(std::make_index_sequence<cols>())
            )
        {
        }
        GGG_CUDA explicit constexpr Matrix(const std::array<Vector<T, rows>, cols>& columns)
            : m_columns(columns)
        {
        }
        ~Matrix() = default;

        [[nodiscard]] GGG_CUDA constexpr Vector<T, rows>& operator[](std::size_t c) { return m_columns[c]; }
        [[nodiscard]] GGG_CUDA constexpr const Vector<T, rows>& operator[](std::size_t c) const { return m_columns[c]; }
        [[nodiscard]] GGG_CUDA constexpr T& operator()(std::size_t c, std::size_t r) { return m_columns[c][r]; }
        [[nodiscard]] GGG_CUDA constexpr const T& operator()(std::size_t c, std::size_t r) const { return m_columns[c][r]; }

        [[nodiscard]] GGG_CUDA constexpr Vector<T, cols> GetRow(std::size_t r) const
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Vector<T, cols>
            {
                return { m_columns[Cs][r]... };
            }(std::make_index_sequence<cols>());
        }

        [[nodiscard]] GGG_CUDA constexpr Vector<T, rows> GetCol(std::size_t c) const
        {
            return m_columns[c];
        }

        [[nodiscard]] friend GGG_CUDA constexpr Matrix operator+(const Matrix& lhs, const Matrix& rhs)
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Matrix
            {
                return { std::array<Vector<T, rows>, cols>{ (lhs.m_columns[Cs] + rhs.m_columns[Cs])... } };
            }(std::make_index_sequence<cols>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Matrix operator+(const Matrix& lhs, const T& rhs)
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Matrix
            {
                return { std::array<Vector<T, rows>, cols>{ (lhs.m_columns[Cs] + rhs)... } };
            }(std::make_index_sequence<cols>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Matrix operator+(const T& lhs, const Matrix& rhs)
        {
            return rhs + lhs;
        }

        GGG_CUDA constexpr Matrix& operator+=(const Matrix& rhs)
        {
            *this = *this + rhs;
            return *this;
        }

        GGG_CUDA constexpr Matrix& operator+=(const T& rhs)
        {
            *this = *this + rhs;
            return *this;
        }

        [[nodiscard]] friend GGG_CUDA constexpr Matrix operator-(const Matrix& lhs, const Matrix& rhs)
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Matrix
            {
                return { std::array<Vector<T, rows>, cols>{ (lhs.m_columns[Cs] - rhs.m_columns[Cs])... } };
            }(std::make_index_sequence<cols>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Matrix operator-(const Matrix& lhs, const T& rhs)
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Matrix
            {
                return { std::array<Vector<T, rows>, cols>{ (lhs.m_columns[Cs] - rhs)... } };
            }(std::make_index_sequence<cols>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Matrix operator-(const T& lhs, const Matrix& rhs)
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Matrix
            {
                return { std::array<Vector<T, rows>, cols>{ (lhs - rhs.m_columns[Cs])... } };
            }(std::make_index_sequence<cols>());
        }

        GGG_CUDA constexpr Matrix& operator-=(const Matrix& rhs)
        {
            *this = *this - rhs;
            return *this;
        }

        GGG_CUDA constexpr Matrix& operator-=(const T& rhs)
        {
            *this = *this - rhs;
            return *this;
        }

        [[nodiscard]] friend GGG_CUDA constexpr Matrix operator*(const Matrix& lhs, const T& rhs)
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Matrix
            {
                return { std::array<Vector<T, rows>, cols>{ (lhs.m_columns[Cs] * rhs)... } };
            }(std::make_index_sequence<cols>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Matrix operator*(const T& lhs, const Matrix& rhs)
        {
            return rhs * lhs;
        }

        GGG_CUDA constexpr Matrix& operator*=(const T& rhs)
        {
            *this = *this * rhs;
            return *this;
        }

        [[nodiscard]] friend GGG_CUDA constexpr Matrix operator/(const Matrix& lhs, const T& rhs)
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Matrix
            {
                return { std::array<Vector<T, rows>, cols>{ (lhs.m_columns[Cs] / rhs)... } };
            }(std::make_index_sequence<cols>());
        }

        GGG_CUDA constexpr Matrix& operator/=(const T& rhs)
        {
            *this = *this / rhs;
            return *this;
        }

        [[nodiscard]] GGG_CUDA constexpr Matrix operator-()
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Matrix
            {
                return { std::array<Vector<T, rows>, cols>{ (-m_columns[Cs])... } };
            }(std::make_index_sequence<cols>());
        }

        template<std::size_t OtherCols> requires (OtherCols > 0)
        [[nodiscard]] friend GGG_CUDA constexpr Matrix<T, rows, OtherCols> operator*(const Matrix& lhs, const Matrix<T, cols, OtherCols>& rhs)
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Matrix<T, rows, OtherCols>
            {
                return { std::array<Vector<T, rows>, OtherCols>{ (lhs * rhs[Cs])... } };
            }(std::make_index_sequence<OtherCols>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector<T, rows> operator*(const Matrix& m, const Vector<T, cols>& v)
        {
            return [&] <std::size_t... Rs> (std::index_sequence<Rs...>) -> Vector<T, rows>
            {
                return { Dot(m.GetRow(Rs), v)... };
            }(std::make_index_sequence<rows>());
        }

        [[nodiscard]] friend GGG_CUDA constexpr Vector<T, cols> operator*(const Vector<T, rows>& v, const Matrix& m)
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Vector<T, cols>
            {
                return { Dot(v, m.GetCol(Cs))... };
            }(std::make_index_sequence<cols>());
        }

        [[nodiscard]] friend constexpr bool operator==(const Matrix& lhs, const Matrix& rhs) = default;
        [[nodiscard]] friend constexpr auto operator<=>(const Matrix& lhs, const Matrix& rhs) = default;

        [[nodiscard]] GGG_CUDA constexpr Matrix<T, cols, rows> Transpose() const
        {
            return [&] <std::size_t... Cs> (std::index_sequence<Cs...>) -> Matrix<T, cols, rows>
            {
                return { std::array<Vector<T, cols>, rows>{ GetRow(Cs)... } };
            }(std::make_index_sequence<rows>());
        }

        [[nodiscard]] GGG_CUDA constexpr Matrix Inverse() const requires (rows == cols && std::floating_point<T>)
        {
            return detail::MatrixInverse<T, rows>::Compute(*this);
        }

        friend std::ostream& operator<<(std::ostream& os, const Matrix& m)
        {
            os << "{";
            for (std::size_t r = 0; r < rows; ++r)
            {
                os << (r == 0 ? "{" : ",{");
                for (std::size_t c = 0; c < cols; ++c)
                {
                    os << (c == 0 ? "" : ",") << m(c, r);
                }
                os << "}";
            }
            os << "}";
            return os;
        }

    private:
        std::array<Vector<T, rows>, cols> m_columns{};
    };

    namespace detail
    {
        template<typename T>
        [[nodiscard]] GGG_CUDA constexpr T Abs(const T& x)
        {
            return x < T{0} ? -x : x;
        }

        template<typename T, std::size_t N>
        struct MatrixInverse
        {
            [[nodiscard]] GGG_CUDA static constexpr Matrix<T, N, N> Compute(const Matrix<T, N, N>& m)
            {
                Matrix<T, N, N> a = m;
                Matrix<T, N, N> inv{};

                for (std::size_t i = 0; i < N; ++i)
                {
                    std::size_t pivotRow = i;
                    T pivotAbs = Abs(a(i, i));
                    for (std::size_t r = i + 1; r < N; ++r)
                    {
                        const T rowAbs = Abs(a(i, r));
                        if (rowAbs > pivotAbs)
                        {
                            pivotAbs = rowAbs;
                            pivotRow = r;
                        }
                    }

                    if (pivotAbs == T{0})
                    {
                        return Matrix<T, N, N>{};
                    }

                    if (pivotRow != i)
                    {
                        for (std::size_t c = 0; c < N; ++c)
                        {
                            std::swap(a(c, i), a(c, pivotRow));
                            std::swap(inv(c, i), inv(c, pivotRow));
                        }
                    }

                    const T pivot = a(i, i);
                    const T invPivot = T{1} / pivot;

                    for (std::size_t c = 0; c < N; ++c)
                    {
                        a(c, i) *= invPivot;
                        inv(c, i) *= invPivot;
                    }

                    for (std::size_t r = 0; r < N; ++r)
                    {
                        if (r == i)
                        {
                            continue;
                        }

                        const T factor = a(i, r);
                        if (factor == T{0})
                        {
                            continue;
                        }

                        for (std::size_t c = 0; c < N; ++c)
                        {
                            a(c, r) -= factor * a(c, i);
                            inv(c, r) -= factor * inv(c, i);
                        }
                    }
                }

                return inv;
            }
        };

        template<typename T>
        struct MatrixInverse<T, 2>
        {
            [[nodiscard]] GGG_CUDA static constexpr Matrix<T, 2, 2> Compute(const Matrix<T, 2, 2>& m)
            {
                const T a00 = m(0, 0);
                const T a01 = m(1, 0);
                const T a10 = m(0, 1);
                const T a11 = m(1, 1);
                const T det = (a00 * a11) - (a01 * a10);
                const T invDet = T{1} / det;

                Matrix<T, 2, 2> inv;
                inv(0, 0) = a11 * invDet;
                inv(1, 0) = -a01 * invDet;
                inv(0, 1) = -a10 * invDet;
                inv(1, 1) = a00 * invDet;
                return inv;
            }
        };

        template<typename T>
        struct MatrixInverse<T, 3>
        {
            [[nodiscard]] GGG_CUDA static constexpr Matrix<T, 3, 3> Compute(const Matrix<T, 3, 3>& m)
            {
                const T a00 = m(0, 0);
                const T a01 = m(1, 0);
                const T a02 = m(2, 0);
                const T a10 = m(0, 1);
                const T a11 = m(1, 1);
                const T a12 = m(2, 1);
                const T a20 = m(0, 2);
                const T a21 = m(1, 2);
                const T a22 = m(2, 2);

                const T c00 = (a11 * a22) - (a12 * a21);
                const T c01 = -((a10 * a22) - (a12 * a20));
                const T c02 = (a10 * a21) - (a11 * a20);

                const T c10 = -((a01 * a22) - (a02 * a21));
                const T c11 = (a00 * a22) - (a02 * a20);
                const T c12 = -((a00 * a21) - (a01 * a20));

                const T c20 = (a01 * a12) - (a02 * a11);
                const T c21 = -((a00 * a12) - (a02 * a10));
                const T c22 = (a00 * a11) - (a01 * a10);

                const T det = (a00 * c00) + (a01 * c01) + (a02 * c02);
                const T invDet = T{1} / det;

                Matrix<T, 3, 3> inv;
                inv(0, 0) = c00 * invDet;
                inv(0, 1) = c01 * invDet;
                inv(0, 2) = c02 * invDet;

                inv(1, 0) = c10 * invDet;
                inv(1, 1) = c11 * invDet;
                inv(1, 2) = c12 * invDet;

                inv(2, 0) = c20 * invDet;
                inv(2, 1) = c21 * invDet;
                inv(2, 2) = c22 * invDet;
                return inv;
            }
        };

        template<typename T>
        struct MatrixInverse<T, 4>
        {
            [[nodiscard]] GGG_CUDA static constexpr Matrix<T, 4, 4> Compute(const Matrix<T, 4, 4>& m)
            {
                const T a00 = m(0, 0);
                const T a01 = m(1, 0);
                const T a02 = m(2, 0);
                const T a03 = m(3, 0);
                const T a10 = m(0, 1);
                const T a11 = m(1, 1);
                const T a12 = m(2, 1);
                const T a13 = m(3, 1);
                const T a20 = m(0, 2);
                const T a21 = m(1, 2);
                const T a22 = m(2, 2);
                const T a23 = m(3, 2);
                const T a30 = m(0, 3);
                const T a31 = m(1, 3);
                const T a32 = m(2, 3);
                const T a33 = m(3, 3);

                const T s0 = (a00 * a11) - (a10 * a01);
                const T s1 = (a00 * a12) - (a10 * a02);
                const T s2 = (a00 * a13) - (a10 * a03);
                const T s3 = (a01 * a12) - (a11 * a02);
                const T s4 = (a01 * a13) - (a11 * a03);
                const T s5 = (a02 * a13) - (a12 * a03);

                const T c5 = (a22 * a33) - (a32 * a23);
                const T c4 = (a21 * a33) - (a31 * a23);
                const T c3 = (a21 * a32) - (a31 * a22);
                const T c2 = (a20 * a33) - (a30 * a23);
                const T c1 = (a20 * a32) - (a30 * a22);
                const T c0 = (a20 * a31) - (a30 * a21);

                const T det = (s0 * c5) - (s1 * c4) + (s2 * c3) + (s3 * c2) - (s4 * c1) + (s5 * c0);
                const T invDet = T{1} / det;

                Matrix<T, 4, 4> inv;
                inv(0, 0) = ((a11 * c5) - (a12 * c4) + (a13 * c3)) * invDet;
                inv(1, 0) = (-(a01 * c5) + (a02 * c4) - (a03 * c3)) * invDet;
                inv(2, 0) = ((a31 * s5) - (a32 * s4) + (a33 * s3)) * invDet;
                inv(3, 0) = (-(a21 * s5) + (a22 * s4) - (a23 * s3)) * invDet;

                inv(0, 1) = (-(a10 * c5) + (a12 * c2) - (a13 * c1)) * invDet;
                inv(1, 1) = ((a00 * c5) - (a02 * c2) + (a03 * c1)) * invDet;
                inv(2, 1) = (-(a30 * s5) + (a32 * s2) - (a33 * s1)) * invDet;
                inv(3, 1) = ((a20 * s5) - (a22 * s2) + (a23 * s1)) * invDet;

                inv(0, 2) = ((a10 * c4) - (a11 * c2) + (a13 * c0)) * invDet;
                inv(1, 2) = (-(a00 * c4) + (a01 * c2) - (a03 * c0)) * invDet;
                inv(2, 2) = ((a30 * s4) - (a31 * s2) + (a33 * s0)) * invDet;
                inv(3, 2) = (-(a20 * s4) + (a21 * s2) - (a23 * s0)) * invDet;

                inv(0, 3) = (-(a10 * c3) + (a11 * c1) - (a12 * c0)) * invDet;
                inv(1, 3) = ((a00 * c3) - (a01 * c1) + (a02 * c0)) * invDet;
                inv(2, 3) = (-(a30 * s3) + (a31 * s1) - (a32 * s0)) * invDet;
                inv(3, 3) = ((a20 * s3) - (a21 * s1) + (a22 * s0)) * invDet;

                return inv;
            }
        };
    }

    using Mat2f = Matrix<float, 2, 2>;
    using Mat3f = Matrix<float, 3, 3>;
    using Mat4f = Matrix<float, 4, 4>;

    using Mat2d = Matrix<double, 2, 2>;
    using Mat3d = Matrix<double, 3, 3>;
    using Mat4d = Matrix<double, 4, 4>;
}

#undef GGG_CUDA
