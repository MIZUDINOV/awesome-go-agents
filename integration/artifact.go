package integration

import "context"

// ArtifactRef identifies complete output retained outside the model surface.
// The core never assumes a storage technology; hosts may map this to an
// object store, database blob, or content-addressed filesystem.
type ArtifactRef struct {
	ID          string `json:"id"`
	Size        int64  `json:"size"`
	SHA256      string `json:"sha256"`
	ContentType string `json:"content_type,omitempty"`
}

// ArtifactStore is an optional durable spill port for complete tool output.
// Implementations must make Put idempotent for the same owner/name/content
// and return a stable reference that can be replayed from the session log.
type ArtifactStore interface {
	Put(context.Context, string, string, []byte, string) (ArtifactRef, error)
}
