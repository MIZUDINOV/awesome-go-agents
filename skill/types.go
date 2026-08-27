// Package skill implements provider-neutral discovery and loading of reusable
// agent instructions. It deliberately owns no product database, UI, network
// transport, or global registry.
package skill

import (
	"context"
	"errors"
	"regexp"
	"time"
	"unicode/utf8"
)

const (
	DefaultCatalogDescriptionMaxLength = 500
	DefaultCollectCacheMaxEntries      = 128
	MaxSkillNameLength                 = 64
	MaxSkillDescriptionLength          = 1024
	MaxSkillCompatibilityLength        = 500
	RuntimeProviderName                = "runtime"
	RuntimeRank                        = 250
	BundledRank                        = 600
)

var (
	ErrInvalidSkill       = errors.New("skill: invalid skill")
	ErrSkillNotFound      = errors.New("skill: not found")
	ErrPolicyDenied       = errors.New("skill: invocation denied")
	ErrIncompleteCatalog  = errors.New("skill: incomplete catalog")
	ErrPinnedMismatch     = errors.New("skill: pinned definition mismatch")
	ErrProviderDisposed   = errors.New("skill: provider disposed")
	ErrDuplicateProvider  = errors.New("skill: duplicate provider")
	ErrInvalidResource    = errors.New("skill: invalid resource")
	ErrSkillNotLoaded     = errors.New("skill: not loaded")
	ErrUnsupportedMutable = errors.New("skill: provider cannot guarantee pinned content")
	ErrAlreadyLoaded      = errors.New("skill: already loaded")
)

var namePattern = regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*$`)

func IsName(value string) bool {
	return utf8.RuneCountInString(value) <= MaxSkillNameLength && namePattern.MatchString(value)
}

type InvocationPolicy struct {
	Model bool `json:"model_invocable"`
	User  bool `json:"user_invocable"`
}

type ResourceKind string

const (
	ResourceDirectory ResourceKind = "directory"
	ResourceURL       ResourceKind = "url"
	ResourceOpaque    ResourceKind = "opaque"
)

type ResourceBase struct {
	Kind        ResourceKind `json:"kind"`
	Path        string       `json:"path,omitempty"`
	URL         string       `json:"url,omitempty"`
	Description string       `json:"description,omitempty"`
}

type ResourceRef struct {
	Name        string `json:"name" yaml:"name"`
	URL         string `json:"url,omitempty" yaml:"url,omitempty"`
	SHA256      string `json:"sha256,omitempty" yaml:"sha256,omitempty"`
	MediaType   string `json:"media_type,omitempty" yaml:"media-type,omitempty"`
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

type Summary struct {
	Name         string           `json:"name"`
	Description  string           `json:"description"`
	WhenToUse    string           `json:"when_to_use,omitempty"`
	Policy       InvocationPolicy `json:"invocation"`
	Provider     string           `json:"provider"`
	Source       string           `json:"source"`
	ResourceBase *ResourceBase    `json:"resource_base,omitempty"`
}

type Candidate struct {
	Summary
	Rank        int            `json:"rank"`
	Locator     string         `json:"locator"`
	Version     string         `json:"version"`
	ContentHash string         `json:"content_hash"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

type Definition struct {
	Summary
	Version          string         `json:"version"`
	ContentHash      string         `json:"content_hash"`
	Content          string         `json:"content"`
	ResourceManifest []ResourceRef  `json:"resources"`
	Metadata         map[string]any `json:"metadata,omitempty"`
}

type Observation struct {
	Candidates []Candidate `json:"candidates"`
	Complete   bool        `json:"complete"`
	Revision   string      `json:"revision,omitempty"`
}

type ListRequest struct {
	CWD    string   `json:"cwd,omitempty"`
	Scopes []string `json:"scopes,omitempty"`
}

type Lookup struct {
	ListRequest
	Name        string `json:"name"`
	Provider    string `json:"provider,omitempty"`
	Scope       string `json:"scope,omitempty"`
	Locator     string `json:"locator,omitempty"`
	Version     string `json:"version,omitempty"`
	ContentHash string `json:"content_hash,omitempty"`
}

type Catalog struct {
	Skills   []Summary `json:"skills"`
	Complete bool      `json:"complete"`
	Hash     string    `json:"hash"`
}

type PinnedSkill struct {
	Candidate
	Scope string `json:"scope,omitempty"`
}

type Snapshot struct {
	SchemaVersion string        `json:"schema_version"`
	Skills        []PinnedSkill `json:"skills"`
	Complete      bool          `json:"complete"`
	CatalogHash   string        `json:"catalog_hash"`
	SnapshotHash  string        `json:"snapshot_hash"`
	CreatedAt     time.Time     `json:"created_at"`
}

type Provider interface {
	List(context.Context, ListRequest) (Observation, error)
	Get(context.Context, Candidate, Lookup) (Definition, error)
}

// PinnedProvider declares whether a provider can resolve an immutable locator
// for the lifetime of a pinned run. Providers that omit this capability remain
// usable for live discovery but fail closed in LoadPinned.
type PinnedProvider interface {
	SupportsPinnedLookup() bool
}

type ProviderOptions struct {
	Name     string
	Scope    string
	Provider Provider
}

type Invalidation struct {
	Provider string
	Scope    string
	Reason   string
}

type Resource struct {
	Name      string
	MediaType string
	Data      []byte
	URL       string
	SHA256    string
}

type ResourceResolver interface {
	Resolve(context.Context, Definition, ResourceRef) (Resource, error)
}

// ResourcePathResolver resolves one provider-owned relative resource after
// its skill has been activated. Implementations must contain the reference
// within the selected skill and must not expose host filesystem locators.
type ResourcePathResolver interface {
	ResolvePath(context.Context, Definition, string) (Resource, error)
}
