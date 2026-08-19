package integration

import (
	"context"
	"errors"
	"io"
)

// Web is the provider-neutral web capability seam (search and fetch).
type Web interface {
	Search(ctx context.Context, query string, opts SearchOptions) ([]SearchResult, error)
	Fetch(ctx context.Context, url string, opts FetchOptions) (*FetchedDocument, error)
}

// SearchOptions bounds a web search request.
type SearchOptions struct {
	// MaxResults caps returned results (default ~8).
	MaxResults int
}

// SearchResult is one web search hit.
type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

// FetchOptions bounds a fetch request.
type FetchOptions struct {
	// MaxBytes caps the fetched body (default ~200k chars).
	MaxBytes int
	// TimeoutSec bounds the request (default 30).
	TimeoutSec int
}

// FetchedDocument is a bounded web page capture.
type FetchedDocument struct {
	URL     string
	Title   string
	Content string
	// Reader streams content > MaxBytes for spill.
	Reader io.Reader
}

// Sentinel web errors.
var (
	ErrFetchTooLarge = errors.New("fetched document exceeds the bounded size")
)
