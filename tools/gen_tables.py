#!/usr/bin/env python3
"""Generate tables_generated.go from lm15-contract tables/providers.json.

The contract file is the reference's provider tables as data (its
tables/README.md): registry rows with their whole access policies, the
managed-login declared providers, the compat presets with their base-URL and
alias tables, the router's built-in rules and litellm prefixes, and the
managed-login service labels. This package reads them from the generated
file; nothing here re-derives a value (playbooks/port.md rule 2). Output is
gofmt-formatted (`gofmt` must be on PATH).

    python3 tools/gen_tables.py [--contract ../lm15-contract] [--check]

`--check` writes nothing and exits 1 when the committed file is stale (CI runs
it against the contract checkout at CONTRACT_PIN). Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tables_generated.go"
SCHEMA = 1
INITIALISMS = {"api": "API", "url": "URL", "json": "JSON", "id": "ID", "sigv4": "SigV4"}


def pascal(name: str) -> str:
    return "".join(INITIALISMS.get(part, part[:1].upper() + part[1:]) for part in name.split("_"))


def s(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # a valid Go interpreted string literal


def strings(values: list[str]) -> str | None:
    return "[]string{" + ", ".join(s(v) for v in values) + "}" if values else None


def pairs(values: list[list[str]]) -> str | None:
    return "[][2]string{" + ", ".join("{" + s(a) + ", " + s(b) + "}" for a, b in values) + "}" if values else None


def string_map(values: dict[str, str]) -> str | None:
    return "map[string]string{" + ", ".join(f"{s(k)}: {s(v)}" for k, v in values.items()) + "}" if values else None


def struct(type_: str, fields: list[tuple[str, str | None]]) -> str:
    body = ", ".join(f"{name}: {value}" for name, value in fields if value is not None)
    return f"{type_}{{{body}}}"


def setting(x: dict) -> str:
    return struct("", [("Name", s(x["name"])), ("Env", strings(x["env"])),
                       ("Default", s(x["default"]) if x["default"] is not None else None)])


def settings(values: list[dict]) -> str | None:
    return "[]HostSetting{" + ", ".join(setting(x) for x in values) + "}" if values else None


def supports(x: dict) -> str:
    fields = [(pascal(k), "true") for k, v in x.items() if k != "extra" and v is True]
    return struct("EndpointSupport", fields + [("Extra", strings(x["extra"]))])


def host(h: dict | None) -> str | None:
    if h is None:
        return None
    return struct("&HostSpec", [
        ("BaseURL", s(h["base_url"])), ("Settings", settings(h["settings"])), ("Paths", string_map(h["paths"])),
        ("ModelIn", s(h["model_in"])), ("AnthropicVersionIn", s(h["anthropic_version_in"])),
        ("StreamFraming", s(h["stream_framing"])), ("RequiredHeaders", pairs(h["required_headers"])),
        ("SigV4Service", s(h["sigv4_service"]) if h["sigv4_service"] is not None else None),
        ("EndpointEnv", strings(h["endpoint_env"])),
    ])


def access(a: dict) -> str:
    opt = lambda v: s(v) if v is not None else None  # noqa: E731
    return struct("AccessPolicy", [
        ("Provider", s(a["provider"])), ("Supports", supports(a["supports"])),
        ("AuthModes", strings(a["auth_modes"])), ("EnterpriseVariants", strings(a["enterprise_variants"])),
        ("EnvKeys", strings(a["env_keys"])), ("CredentialPolicy", s(a["credential_policy"])),
        ("AuthScheme", strings(a["auth_scheme"])), ("Headers", pairs(a["headers"])), ("Host", host(a["host"])),
        ("LoginHint", opt(a["login_hint"])), ("Backend", s(a["backend"])),
        ("BackendOptions", string_map(a["backend_options"])), ("SystemPrefix", opt(a["system_prefix"])),
        ("BaseURL", opt(a["base_url"])), ("BackendSettings", settings(a["backend_settings"])),
    ])


def knob_value(value) -> str:
    # Go carries an override knob as one string; a list knob is comma-joined
    # (compat.go ForModel splits reasoning_efforts on ",").
    return ",".join(value) if isinstance(value, list) else value


def compat(type_: str, c: dict) -> str:
    fields = []
    for k, v in c.items():
        if k == "model_overrides":
            overrides = ", ".join(
                "{Prefix: " + s(prefix) + ", Knobs: " + string_map({kk: knob_value(kv) for kk, kv in knobs.items()}) + "}"
                for prefix, knobs in v)
            fields.append(("ModelOverrides", "[]ModelOverride{" + overrides + "}"))
        elif isinstance(v, list):
            fields.append((pascal(k), strings(v) or "[]string{}"))
        elif isinstance(v, str):
            fields.append((pascal(k), s(v)))
        else:
            raise SystemExit(f"{type_}.{k}: no Go form for {v!r} (extend tools/gen_tables.py)")
    return struct(type_, fields)


def row(r: dict) -> str:
    c = r["compat"]
    compat_name, compat_value = None, None
    if isinstance(c, str):
        compat_name = s(c)
    elif isinstance(c, dict) and c:
        if r["dialect"] != "openai-chat":
            raise SystemExit(f"{r['id']}: a {r['dialect']} compat object has no Go field (ProviderDefinition.CompatValue is *OpenAIChatCompat)")
        compat_value = compat("&OpenAIChatCompat", c)
    return struct("", [
        ("ID", s(r["id"])), ("Dialect", s(r["dialect"])), ("Kind", s(r["kind"])),
        ("Compat", compat_name), ("CompatValue", compat_value), ("Access", access(r["access"])),
        ("Aliases", strings(r["aliases"])),
        ("PlaceholderKey", s(r["placeholder_key"]) if r["placeholder_key"] is not None else None),
        ("ConsoleURL", s(r["console_url"]) if r["console_url"] is not None else None),
        ("Note", s(r["note"])),
    ])


def render(tables: dict, digest: str) -> str:
    if tables.get("schema") != SCHEMA:
        raise SystemExit(f"tables/providers.json schema {tables.get('schema')!r}; this generator reads {SCHEMA}")
    c = tables["compat"]
    rules = ", ".join("{" + s(r["prefix"]) + ", " + s(r["provider"]) + ", " + s(r["note"]) + "}"
                      for r in tables["routing"]["default_rules"])
    presets = lambda type_, table: "map[string]" + type_ + "{\n" + "".join(  # noqa: E731
        f"{s(k)}: {compat(type_, v)},\n" for k, v in table.items()) + "}"
    lines = [
        "// Code generated by tools/gen_tables.py from lm15-contract tables/providers.json. DO NOT EDIT.",
        f"// Contract tables sha256 {digest}. Regenerate: python3 tools/gen_tables.py",
        "// The receipts behind each value are cited at the reference's own table",
        "// (lm15-python lm15/registry.py, access.py, compat.py, router.py).",
        "",
        "package lm15",
        "",
        "// tableRow is one provider row of the reference's registry, as data.",
        "type tableRow struct {",
        "ID, Dialect, Kind, Compat string",
        "CompatValue *OpenAIChatCompat",
        "Access AccessPolicy",
        "Aliases []string",
        "PlaceholderKey, ConsoleURL, Note string",
        "}",
        "",
        "// tableProviders are the registry rows in declaration (presentation) order.",
        "var tableProviders = []tableRow{\n" + "".join(row(r) + ",\n" for r in tables["providers"]) + "}",
        "",
        "// tableDeclaredLogin are the managed-login declared providers (AUTH-26: no registry row).",
        "var tableDeclaredLogin = []tableRow{\n" + "".join(row(r) + ",\n" for r in tables["declared_login"]) + "}",
        "",
        "var tableOpenAIChatPresets = " + presets("OpenAIChatCompat", c["chat"]),
        "",
        "var tableOpenAIChatPresetBaseURLs = " + (string_map(c["chat_base_urls"]) or "map[string]string{}"),
        "",
        "var tableOpenAIResponsesPresets = " + presets("OpenAIResponsesCompat", c["responses"]),
        "",
        "var tableOpenAIResponsesPresetBaseURLs = " + (string_map(c["responses_base_urls"]) or "map[string]string{}"),
        "",
        "var tableAnthropicPresets = " + presets("AnthropicCompat", c["anthropic"]),
        "",
        "var tableAnthropicPresetBaseURLs = " + (string_map(c["anthropic_base_urls"]) or "map[string]string{}"),
        "",
        "// tablePresetAliases are read after lowercasing and mapping -, . and spaces to _.",
        "var tablePresetAliases = " + (string_map(c["preset_aliases"]) or "map[string]string{}"),
        "",
        "// tableDefaultRules are the router's built-in prefix rules; first match wins.",
        "var tableDefaultRules = []RouteRule{" + rules + "}",
        "",
        "// tableLitellmPrefixes are litellm's provider/ spellings for the OpenAI-SDK/litellm door.",
        "var tableLitellmPrefixes = " + (string_map(tables["routing"]["litellm_prefixes"]) or "map[string]string{}"),
        "",
        "// tableServiceLabels are the AUTH-12 service labels.",
        "var tableServiceLabels = " + (string_map(tables["login"]["service_labels"]) or "map[string]string{}"),
        "",
    ]
    source = "\n".join(lines)
    proc = subprocess.run(["gofmt"], input=source, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"gofmt rejected the generated source:\n{proc.stderr}")
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--contract", type=Path, default=ROOT.parent / "lm15-contract")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    source = (args.contract / "tables" / "providers.json").read_bytes()
    text = render(json.loads(source.decode("utf-8")), hashlib.sha256(source).hexdigest())
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else None
        if current != text:
            print(f"{OUT.name} is stale for this contract checkout: run python3 tools/gen_tables.py")
            return 1
        print(f"{OUT.name}: current")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
