# Security policy

Feature and match caches currently use Python pickle. Pickle is intended only
for trusted local artifacts and can execute code during deserialization. Never
open a cache downloaded from an unknown source.

Image decoders and pretrained model dependencies should be kept current. Treat
all user-supplied images as untrusted input and process them in an isolated
environment when provenance is uncertain.

Report vulnerabilities privately to the repository owner. Do not include
exploit payloads or sensitive local files in public issues.
