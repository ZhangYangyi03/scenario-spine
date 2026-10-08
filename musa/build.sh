#!/usr/bin/env bash
# Build the MUSA predictor kernel and its driver.
#
# The arch is detected from the device rather than hard-coded: hard-coding an arch
# is how a kernel silently stops matching the silicon it is meant for.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
# The S4000 reports compute capability 2.2, so the default target is mp_22 -- NOT
# mp_21. A cubin built for the wrong minor version loads and then fails at launch
# with a bare error code (98, invalid device function), which is a five-minute
# detour every time; the arch is therefore derived from the device rather than
# guessed, and MUSA_ARCH exists only to override it.
ARCH="${MUSA_ARCH:-}"

if [ -z "$ARCH" ]; then
  ARCH="$(python3 - <<'PY' 2>/dev/null || echo mp_21
import torch
try:
    import torch_musa
    p = torch.musa.get_device_properties(0)
    major = p.major.strip(",") if isinstance(p.major, str) else p.major
    print(f"mp_{int(major)}{int(p.minor)}")
except Exception:
    print("mp_21")
PY
)"
fi

echo "mcc: $("$MUSA_HOME/bin/mcc" --version | head -1)"
echo "arch: $ARCH"

"$MUSA_HOME/bin/mcc" -c -x musa -O3 -fPIC --offload-arch="$ARCH" \
  -I"$MUSA_HOME/include" "$HERE/predict.mu" -o "$HERE/predict.o"

g++ -O2 -std=c++17 -fPIC -no-pie -I"$MUSA_HOME/include" -L"$MUSA_HOME/lib" \
  "$HERE/driver.cpp" "$HERE/predict.o" -o "$HERE/predict" \
  -lmusa -lmusart -lpthread -Wl,-rpath,"$MUSA_HOME/lib"

echo "built $HERE/predict"
