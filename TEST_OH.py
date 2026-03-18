import oh_group
import numpy as np
from collections import Counter

def main():
    print("=" * 60)
    print("CALCULATION OF THE AVERAGE OF (Tr(R)²/3) FOR THE O_h GROUP")
    print("=" * 60)

    # Call the Rust kernel
    average = oh_group.calcular_promedio_oh()

    print("\n" + "=" * 60)
    print(f"FINAL RESULT: AVERAGE = {average:.6f}")
    print("=" * 60)

    # Theoretical verification
    print("\n=== THEORETICAL VERIFICATION ===")
    print(f"Obtained value:         {average:.6f}")
    print(f"Expected value (0.4125): {0.4125:.6f}")
    print(f"Difference:              {abs(average - 0.4125):.6f}")

    if abs(average - 0.4125) < 1e-4:
        print("\n✅ VERIFIED: The average is 0.4125 (within numerical error)")
    else:
        print("\n❌ DISCREPANCY: The value is not 0.4125")

if __name__ == "__main__":
    main()