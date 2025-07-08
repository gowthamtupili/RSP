#ifndef TARGET_TYPES_CUH
#define TARGET_TYPES_CUH

namespace TargetProcessing {
    struct CUDATarget {
        double x, y, z;
        double range;
        double azimuth;
        double elevation;
        double strength;
        double rcs;
        double relativeSpeed;
    };
}

#endif // TARGET_TYPES_CUH
