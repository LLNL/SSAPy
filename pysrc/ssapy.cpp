#include "ssapy.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace ssapy {
    void pyExportSSAPy(py::module& m) {
        py::class_<Ellipsoid, std::shared_ptr<Ellipsoid>>(m, "Ellipsoid")
            .def(py::init<const double, const double>(),
                "Req"_a=6378.137e3, "f"_a=1.0/298.257223563
            )
            .def("_sphereToCart",
                [](
                    const Ellipsoid& e,
                    size_t lonarr, size_t latarr, size_t heightarr, size_t size,
                    size_t xarr, size_t yarr, size_t zarr
                ) {
                    double* lonptr = reinterpret_cast<double*>(lonarr);
                    double* latptr = reinterpret_cast<double*>(latarr);
                    double* heightptr = reinterpret_cast<double*>(heightarr);
                    double* xptr = reinterpret_cast<double*>(xarr);
                    double* yptr = reinterpret_cast<double*>(yarr);
                    double* zptr = reinterpret_cast<double*>(zarr);
                    for(int i=0; i<size; i++) {
                        e.sphereToCart(
                            lonptr[i], latptr[i], heightptr[i],
                            xptr[i], yptr[i], zptr[i]
                        );
                    }
                }
            )
            .def("_cartToSphere",
                [](
                    const Ellipsoid& e,
                    size_t xarr, size_t yarr, size_t zarr, size_t size,
                    size_t lonarr, size_t latarr, size_t heightarr
                ) {
                    double* xptr = reinterpret_cast<double*>(xarr);
                    double* yptr = reinterpret_cast<double*>(yarr);
                    double* zptr = reinterpret_cast<double*>(zarr);
                    double* lonptr = reinterpret_cast<double*>(lonarr);
                    double* latptr = reinterpret_cast<double*>(latarr);
                    double* heightptr = reinterpret_cast<double*>(heightarr);
                    for(int i=0; i<size; i++) {
                        e.cartToSphere(
                            xptr[i], yptr[i], zptr[i],
                            lonptr[i], latptr[i], heightptr[i]
                        );
                    }
                }
            );

        py::class_<HarrisPriester>(m, "HarrisPriester")
            .def(py::init<const Ellipsoid&, const double>(),
                "ellip"_a, "n"_a=3.0, py::keep_alive<1, 2>()
            )
            .def("density", &HarrisPriester::density);

        py::class_<AccelHarmonic>(m, "AccelHarmonic")
            .def(py::init(
                [](const double GM, const double R, const int ncol, size_t CSptr) {
                    return new AccelHarmonic(GM, R, ncol, reinterpret_cast<double*>(CSptr));
                }
            ))
            .def("accel",
                [](
                    const AccelHarmonic& ah,
                    const int n_max, const int m_max,
                    size_t inptr, size_t outptr
                ){
                    double* in = reinterpret_cast<double*>(inptr);
                    double* out = reinterpret_cast<double*>(outptr);
                    ah.accel(in[0], in[1], in[2], n_max, m_max, out[0], out[1], out[2]);
                }
            );
    }

    PYBIND11_MODULE(_ssapy, m) {
        pyExportSSAPy(m);
    }
}
