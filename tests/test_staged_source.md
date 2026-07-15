# `test_staged_source.py`

## Purpose

Protects exact tensor delivery and accounting for the B2.3 asynchronous host
staging source.

## Coverage

- serves a predicted page from completed background staging; and
- sends an unpredicted page through the ordinary direct source path.
