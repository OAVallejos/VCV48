# test_ohv4.py
import analysis_kernel

def main():
    print("=" * 60)
    print("Test of the analysis_kernel kernel (Kernel 4/5)")
    print("=" * 60)
    print()

    # Complete report
    print(analysis_kernel.report_py())

    # Test of stabilize
    print("\n=== Test of stabilize ===")
    s = analysis_kernel.stabilize(0.2833, 0.2833, 0.9162)
    print(f"  stabilize(0.2833, 0.2833, 0.9162) = {s} (expected: 2)")
    assert s == 2

    s = analysis_kernel.stabilize(1.0, 0.0, 0.0)
    print(f"  stabilize(1, 0, 0) = {s} (expected: 8)")
    assert s == 8

    s = analysis_kernel.stabilize(0.3827, 0.0, 0.9239)
    print(f"  stabilize(0.3827, 0, 0.9239) = {s} (expected: 2)")
    assert s == 2

    # Test of orbit_size
    print("\n=== Test of orbit_size ===")
    o = analysis_kernel.orbit_size(0.2833, 0.2833, 0.9162)
    print(f"  orbit_size(0.2833, 0.2833, 0.9162) = {o} (expected: 24)")
    assert o == 24

    o = analysis_kernel.orbit_size(1.0, 0.0, 0.0)
    print(f"  orbit_size(1, 0, 0) = {o} (expected: 6)")
    assert o == 6

    # Test of classify
    print("\n=== Test of classify ===")
    class_name, eigs = analysis_kernel.classify(0.2833, 0.2833, 0.9162, 0.3285)
    print(f"  Classification of the maximum: {class_name}")
    print(f"  Eigenvalues: {eigs}")
    assert class_name == "maximo"

    class_name, eigs = analysis_kernel.classify(0.3827, 0.0, 0.9239, 0.3125)
    print(f"  Classification of the new point: {class_name}")
    print(f"  Eigenvalues: {eigs}")
    assert class_name == "punto_de_silla"

    print()
    print("=" * 60)
    print("ALL TESTS OF KERNEL 4/5 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()