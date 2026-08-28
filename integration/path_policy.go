package integration

import "strings"

var protectedWorkspaceSearchExcludes = []string{
	"!.git/**", "!**/.git/**", "!.ssh/**", "!**/.ssh/**", "!.aws/**", "!**/.aws/**",
	"!.gcloud/**", "!**/.gcloud/**", "!.kube/**", "!**/.kube/**", "!.wzhooh/**", "!**/.wzhooh/**",
	"!.env*", "!**/.env*", "!.agentkit-*", "!**/.agentkit-*",
	"!.npmrc", "!**/.npmrc", "!.pnpmfile.cjs", "!**/.pnpmfile.cjs", "!.yarnrc*", "!**/.yarnrc*",
	"!.netrc", "!**/.netrc", "!.pypirc", "!**/.pypirc", "!*.pem", "!**/*.pem", "!*.key", "!**/*.key",
}

// IsProtectedWorkspacePath reports whether a workspace-relative path names
// agent state, VCS metadata, credentials, or common private-key material.
// Providers call this before I/O so local and remote execution worlds enforce
// the same model-facing policy.
func IsProtectedWorkspacePath(value string) bool {
	normalized := strings.ReplaceAll(strings.TrimSpace(value), "\\", "/")
	for _, rawSegment := range strings.Split(normalized, "/") {
		segment := strings.ToLower(rawSegment)
		switch segment {
		case ".git", ".ssh", ".aws", ".gcloud", ".kube", ".npmrc", ".pnpmfile.cjs", ".yarnrc", ".yarnrc.yml", ".netrc", ".pypirc", ".wzhooh":
			return true
		}
		if strings.HasPrefix(segment, ".env") || strings.HasPrefix(segment, ".agentkit-") || strings.HasSuffix(segment, ".pem") || strings.HasSuffix(segment, ".key") {
			return true
		}
	}
	return false
}

// ProtectedWorkspaceSearchExcludes returns rg-compatible exclusion globs for
// the same protected path policy. A copy prevents callers from mutating the
// process-wide policy.
func ProtectedWorkspaceSearchExcludes() []string {
	return append([]string(nil), protectedWorkspaceSearchExcludes...)
}
