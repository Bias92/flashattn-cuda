"""Compare unchanged device specializations across the layout enum refactor."""
import hashlib
import json
import re
import subprocess
import sys

CUOBJDUMP = "/usr/local/cuda-12.8/bin/cuobjdump"


def name_key(name):
    # The original bool selected strided (0) or flat contiguous heads (1).
    match = re.match(r"(_Z20attention_fwd_kernelILi\d+E)Lb([01])E(.*)", name)
    if match:
        layout = "0" if match[2] == "0" else "2"
        name = match[1] + "L13AddressLayout" + layout + "E" + match[3]
    # The added enum changes mangler substitution indices in the argument suffix.
    # Geometry, layout, all boolean flags and KV source uniquely identify a kernel.
    return re.match(r"(.+?E7(?:DenseKV|PagedKV))J", name)[1]


def read_functions(path):
    text = subprocess.check_output([CUOBJDUMP, "--dump-sass", path], text=True)
    functions = {}
    current = None
    for line in text.splitlines():
        match = re.search(r"Function : (\S+)", line)
        if match:
            current = name_key(match[1])
            assert current not in functions, current
            functions[current] = []
        elif current and re.match(r"\s*/\*[0-9a-f]+\*/", line):
            functions[current].append(line.strip())
    assert functions and all(functions.values()), path
    return {key: hashlib.sha256("\n".join(lines).encode()).hexdigest()
            for key, lines in functions.items()}


if __name__ == "__main__":
    before, after = map(read_functions, sys.argv[1:3])
    common = sorted(before.keys() & after.keys())
    report = {"before_so": sys.argv[1], "after_so": sys.argv[2],
              "before_count": len(before), "after_count": len(after),
              "unchanged_count": sum(before[k] == after[k] for k in common),
              "changed": [k for k in common if before[k] != after[k]],
              "removed": sorted(before.keys() - after.keys()),
              "added": sorted(after.keys() - before.keys()),
              "after_instruction_sha256": after}
    print(json.dumps(report, indent=2))
