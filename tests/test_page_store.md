# Page-store tests

`test_page_store.py` checks the B2.4 fixed-offset storage contract without a large
checkpoint:

- a tiny layer-major file resolves the requested page and reconstructs exact
  `gate_up` and `down` tensor shapes and values;
- source statistics count one complete page;
- invalid layer or expert coordinates fail before access;
- truncated data and inconsistent metadata are rejected during initialization;
  and
- checkpoint identity hashes prevent silently attaching a same-shape store to
  another checkpoint.

The production builder is exercised on the real checkpoint because constructing
packed safetensor fixtures would duplicate the safetensors library's own format
tests. Its resumability and exact-size invariants are enforced by the same metadata
geometry used in these unit tests.
