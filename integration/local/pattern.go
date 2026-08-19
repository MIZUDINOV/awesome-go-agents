package local

import "regexp"

// compilePattern compiles a grep pattern with bounded complexity. RE2 (Go's
// regexp) is linear-time, so patterns cannot cause catastrophic backtracking.
func compilePattern(pattern string) (*regexp.Regexp, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, err
	}
	return re, nil
}
