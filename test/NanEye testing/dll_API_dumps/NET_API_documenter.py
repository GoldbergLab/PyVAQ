import argparse, os, sys, re, html
import clr
from System.Reflection import Assembly, BindingFlags

# -------------------------------
# CLI args
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Dump .NET APIs from all DLLs in a folder into one HTML with docs and cross-links."
    )
    ap.add_argument("folder", help="Folder containing .dll files")
    ap.add_argument("-o", "--out", default="api.html", help="Output HTML path (default: api.html)")
    ap.add_argument("--all", action="store_true", help="Include non-public types and members")
    ap.add_argument("--recurse", action="store_true", help="Scan subfolders for DLLs")
    return ap.parse_args()

# -------------------------------
# Basic utilities
# -------------------------------
def safe(s):
    return re.sub(r"[^A-Za-z0-9_\-\.]", "-", s or "")

def h(s):
    return html.escape("" if s is None else str(s), quote=True)

def join_comma(seq):
    return ", ".join([str(x) for x in seq if x is not None and str(x) != ""])

# -------------------------------
# Reflection helpers / signatures
# -------------------------------
def flags(public_only):
    f = BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly
    f |= BindingFlags.Public
    if not public_only:
        f |= BindingFlags.NonPublic
    return f

def type_kind(t):
    import System
    if t.IsInterface: return "interface"
    if t.IsEnum: return "enum"
    if t.IsSubclassOf(System.MulticastDelegate): return "delegate"
    if t.IsValueType: return "struct"
    return "class"

def vis_from_methodbase(m):
    if m.IsPublic: return "public"
    if getattr(m, "IsFamilyAndAssembly", False): return "private-protected"
    if getattr(m, "IsFamilyOrAssembly", False): return "protected-internal"
    if m.IsFamily: return "protected"
    if m.IsAssembly: return "internal"
    return "private"

def vis_from_field(f):
    if f.IsPublic: return "public"
    if getattr(f, "IsFamilyAndAssembly", False): return "private-protected"
    if getattr(f, "IsFamilyOrAssembly", False): return "protected-internal"
    if f.IsFamily: return "protected"
    if f.IsAssembly: return "internal"
    return "private"

def vis_from_property(p):
    gm = p.GetGetMethod(True)
    sm = p.GetSetMethod(True)
    order = ["private","private-protected","internal","protected","protected-internal","public"]
    best = None
    for m in (gm, sm):
        if m is None: continue
        v = vis_from_methodbase(m)
        if best is None or order.index(v) > order.index(best):
            best = v
    return best or "private"

def friendly_type(t):
    """Pretty C#-ish name for display (with generics angle brackets)."""
    if t is None: return "void"
    if t.IsByRef: return friendly_type(t.GetElementType()) + "&"
    if t.IsPointer: return friendly_type(t.GetElementType()) + "*"
    if t.IsArray:
        et = t.GetElementType()
        rank = t.GetArrayRank()
        return friendly_type(et) + ("[]" if rank == 1 else "[" + (","*(rank-1)) + "]")
    name = (t.FullName or t.Name or "").replace("+", ".")
    if t.IsGenericType:
        base = name.split('`', 1)[0]
        args = ",".join(friendly_type(a) for a in t.GetGenericArguments())
        return f"{base}<{args}>"
    return name or str(t)

# -------------------------------
# Documentation ID construction
# -------------------------------
def doc_name_for_type_in_id(t, nested_sep='.'):
    if t.IsByRef:   return doc_name_for_type_in_id(t.GetElementType(), nested_sep) + "@"
    if t.IsPointer: return doc_name_for_type_in_id(t.GetElementType(), nested_sep) + "*"
    if t.IsArray:
        et = t.GetElementType()
        rank = t.GetArrayRank()
        return doc_name_for_type_in_id(et, nested_sep) + ("[]" if rank == 1 else "[" + (","*(rank-1)) + "]")
    if t.IsGenericParameter:
        if t.DeclaringMethod is not None:
            return "``" + str(t.GenericParameterPosition)
        return "`" + str(t.GenericParameterPosition)
    if not t.IsGenericType:
        name = (t.FullName or t.Name or "")
        if nested_sep == '.': name = name.replace('+','.')
        return name
    gtd = t.GetGenericTypeDefinition()
    base = (gtd.FullName or gtd.Name or "").split('`',1)[0]
    if nested_sep == '.': base = base.replace('+','.')
    args = ",".join(doc_name_for_type_in_id(a, nested_sep) for a in t.GetGenericArguments())
    return f"{base}{{{args}}}"

def declaring_type_name_for_id(t, nested_sep='.'):
    name = (t.FullName or t.Name or "")
    if nested_sep == '.': name = name.replace('+','.')
    return name

def method_arity_suffix(m):
    try:
        n = 0
        for a in m.GetGenericArguments():
            if a.DeclaringMethod is not None: n += 1
        if n == 0 and m.IsGenericMethod:
            n = len(m.GetGenericArguments())
        return "" if n == 0 else "``" + str(n)
    except:
        return ""

def id_type(t, sep='.'):   return "T:" + declaring_type_name_for_id(t, sep)
def id_ctor(c, sep='.'):
    t = c.DeclaringType
    ctor_name = "#cctor" if c.IsStatic else "#ctor"
    ps = ",".join(doc_name_for_type_in_id(p.ParameterType, sep) for p in c.GetParameters())
    return f"M:{declaring_type_name_for_id(t, sep)}.{ctor_name}({ps})"
def id_method(m, sep='.'):
    t = m.DeclaringType
    name = m.Name + method_arity_suffix(m)
    ps = ",".join(doc_name_for_type_in_id(p.ParameterType, sep) for p in m.GetParameters())
    return f"M:{declaring_type_name_for_id(t, sep)}.{name}({ps})"
def id_property(p, sep='.'):
    t = p.DeclaringType
    name = p.Name
    idx = p.GetIndexParameters()
    if idx is not None and idx.Length > 0:
        plist = ",".join(doc_name_for_type_in_id(ip.ParameterType, sep) for ip in idx)
        return f"P:{declaring_type_name_for_id(t, sep)}.{name}({plist})"
    return f"P:{declaring_type_name_for_id(t, sep)}.{name}"
def id_field(f, sep='.'):
    t = f.DeclaringType
    return f"F:{declaring_type_name_for_id(t, sep)}.{f.Name}"
def id_event(e, sep='.'):
    t = e.DeclaringType
    return f"E:{declaring_type_name_for_id(t, sep)}.{e.Name}"

def id_candidates(builder, *args):
    # try '.' and '+' variants for nested types
    return [builder(*args, sep='.'), builder(*args, sep='+')]

def anchor_from_docid(docid, asmname=None):
    """Turn a doc ID into a safe anchor id; include assembly to avoid collisions."""
    prefix = (asmname + "-") if asmname else ""
    return safe(prefix + docid.replace(":", "-"))

# -------------------------------
# Load XML doc comments
# -------------------------------
from xml.dom import minidom as DOM

def load_doc_map(path):
    """Returns dict[name->memberElement] or {}."""
    if not path or not os.path.exists(path):
        return {}
    try:
        xmldoc = DOM.parse(path)
        out = {}
        for n in xmldoc.getElementsByTagName("member"):
            name = n.getAttribute("name")
            if name:
                out[name] = n
        return out
    except Exception:
        return {}

# -------------------------------
# HTML building helpers
# -------------------------------
def html_tag(tag, attrs=None, inner=""):
    a = ""
    if attrs:
        parts = []
        for k,v in attrs.items():
            if v is None: continue
            parts.append(f'{k}="{h(v)}"')
        if parts: a = " " + " ".join(parts)
    return f"<{tag}{a}>{inner}</{tag}>"

def html_self(tag, attrs=None):
    a = ""
    if attrs:
        parts = []
        for k,v in attrs.items():
            if v is None: continue
            parts.append(f'{k}="{h(v)}"')
        if parts: a = " " + " ".join(parts)
    return f"<{tag}{a}/>"

def link_or_text(text, href):
    if href:
        return f'<a href="{h(href)}">{h(text)}</a>'
    return h(text)

# -------------------------------
# Render doc XML -> HTML (basic)
# -------------------------------
def render_doc_node(node, docid_to_anchor):
    """Recursively render C# XML doc nodes to HTML string."""
    if node.nodeType == node.TEXT_NODE:
        return h(node.data)
    if node.nodeType != node.ELEMENT_NODE:
        return ""

    name = node.tagName
    # Map known tags
    if name in ("summary","remarks","para"):
        inner = "".join(render_doc_node(c, docid_to_anchor) for c in node.childNodes)
        return f"<p>{inner}</p>"
    if name == "code":
        inner = "".join(render_doc_node(c, docid_to_anchor) for c in node.childNodes)
        return f"<pre><code>{inner}</code></pre>"
    if name == "c":
        inner = "".join(render_doc_node(c, docid_to_anchor) for c in node.childNodes)
        return f"<code>{inner}</code>"
    if name in ("param","typeparam","returns","value","exception","example"):
        # label + contents
        inner = "".join(render_doc_node(c, docid_to_anchor) for c in node.childNodes)
        label = name
        if node.hasAttribute("name"):
            label += f' <code>{h(node.getAttribute("name"))}</code>'
        return f"<p><span class='muted small'>{h(label)}:</span> {inner}</p>"
    if name in ("see","seealso"):
        cref = node.getAttribute("cref") if node.hasAttribute("cref") else ""
        text = "".join(render_doc_node(c, docid_to_anchor) for c in node.childNodes).strip()
        if not text:
            # show last identifier segment
            text = cref.split(":")[-1]
        href = None
        if cref in docid_to_anchor:
            href = "#" + docid_to_anchor[cref]
        return link_or_text(text, href)
    if name == "list":
        ltype = node.getAttribute("type") if node.hasAttribute("type") else "bullet"
        items = [c for c in node.childNodes if getattr(c, "tagName", None) == "item"]
        if ltype == "number": tag = "ol"
        else: tag = "ul"
        inner_items = []
        for it in items:
            contents = "".join(render_doc_node(c, docid_to_anchor) for c in it.childNodes)
            inner_items.append(f"<li>{contents}</li>")
        return f"<{tag}>" + "".join(inner_items) + f"</{tag}>"
    # default: render children inline
    inner = "".join(render_doc_node(c, docid_to_anchor) for c in node.childNodes)
    return inner

def render_doc(member_xml_node, docid_to_anchor):
    if member_xml_node is None: return ""
    inner = "".join(render_doc_node(c, docid_to_anchor) for c in member_xml_node.childNodes)
    if inner.strip():
        return f"<blockquote>{inner}</blockquote>"
    return ""

# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print("Folder not found:", folder)
        sys.exit(2)

    # Collect DLLs
    dlls = []
    if args.recurse:
        for root, dirs, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(".dll"):
                    dlls.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(folder):
            if fn.lower().endswith(".dll"):
                dlls.append(os.path.join(folder, fn))
    if not dlls:
        print("No DLLs found.")
        sys.exit(1)

    # Help native/managed deps resolution
    dll_dirs = sorted(set(os.path.dirname(p) for p in dlls))
    for d in dll_dirs:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
        sys.path.append(d)

    # Load assemblies
    assemblies = []
    for p in dlls:
        try:
            asm = Assembly.LoadFrom(p)
            assemblies.append(asm)
        except Exception as ex:
            # Skip native or incompatible DLLs
            print(f"[warn] Skipping {p}: {ex}")
            continue

    # Build doc maps
    doc_maps = {}
    for asm in assemblies:
        loc = asm.Location
        base, _ = os.path.splitext(loc)
        xml_path = base + ".xml"
        m = load_doc_map(xml_path)
        if m:
            doc_maps[asm.FullName] = m

    bf = flags(public_only=not args.all)

    # Gather types and create anchors
    types = []  # list of (asm, type)
    for asm in assemblies:
        try:
            ts = asm.GetExportedTypes() if not args.all else asm.GetTypes()
            for t in ts:
                types.append((asm, t))
        except Exception:
            pass

    # Sort by namespace then name
    types.sort(key=lambda at: ((at[1].Namespace or ""), at[1].Name))

    # Maps for link resolution
    docid_to_anchor = {}
    type_anchor = {}  # (asm.FullName, t) -> anchor id
    type_fqn_to_anchor = {}  # full name string -> anchor (first seen), for best-effort linking

    # Pre-assign type anchors and docid map for types
    for asm, t in types:
        asmname = asm.GetName().Name
        tdoc = id_type(t, sep='.')
        anchor = anchor_from_docid(tdoc, asmname=asmname)
        type_anchor[(asm.FullName, t)] = anchor
        docid_to_anchor[tdoc] = anchor  # T:...
        # also add '+' variant for nested fallback
        tdoc_plus = id_type(t, sep='+')
        docid_to_anchor[tdoc_plus] = anchor
        # for best-effort lookups by friendly name:
        fqn = (t.FullName or t.Name or "").replace("+",".")
        if fqn not in type_fqn_to_anchor:
            type_fqn_to_anchor[fqn] = anchor

    # Helpers to link types/members
    def link_type(t):
        """Return HTML for a type reference, linked if we have an anchor."""
        target_asm = t.Assembly.FullName if hasattr(t, "Assembly") else None
        key = (target_asm, t)
        label = friendly_type(t)
        # Prefer exact match by declaring assembly
        if key in type_anchor:
            return f'<a href="#{type_anchor[key]}">{h(label)}</a>'
        # Fallback by FQN (if unique/available)
        fqn = (t.FullName or t.Name or "").replace("+",".")
        if fqn in type_fqn_to_anchor:
            return f'<a href="#{type_fqn_to_anchor[fqn]}">{h(label)}</a>'
        return h(label)

    def link_docid(docid):
        if docid in docid_to_anchor:
            return "#" + docid_to_anchor[docid]
        return None

    # Start building HTML
    parts = []
    parts.append("""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>.NET API Browser</title>
<meta name="color-scheme" content="light dark">
<style>
  html { scroll-behavior: smooth; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Arial, sans-serif; margin: 1.5rem; }
  h1, h2 { margin: 0.2rem 0; }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
  .masthead { position: sticky; top: 0; z-index: 1000; background: #fff; border-bottom: 1px solid #e5e5e5; padding: 0.5rem 0; }
  .masthead.is-stuck { box-shadow: 0 2px 10px rgba(0,0,0,.06); }
  .asm-header { margin-bottom: 0.25rem; }
  .pill { background:#eee; padding:2px 6px; border-radius: 999px; margin-left:6px; font-size:0.75em; display:inline-block; }
  .muted { color:#666; }
  .small { font-size:0.9em; }
  .toolbar { margin: 0.5rem 0 0.25rem; display:flex; gap:0.5rem; flex-wrap:wrap; }
  .toolbar button { padding:6px 10px; border:1px solid #ccc; border-radius:8px; background:#fff; cursor:pointer; }
  .toolbar button:hover { background:#f3f3f3; }
  .search { margin-left:auto; }
  .search input { padding:6px 10px; border:1px solid #ccc; border-radius:8px; min-width: 240px; }
  details { border: 1px solid #ddd; border-radius: 8px; padding: 0.5rem 0.75rem; margin: 0.5rem 0; background: #fafafa; }
  summary { cursor: pointer; font-weight: 600; }
  .kind { text-transform: uppercase; font-size: 0.75em; color: #555; margin-right: 0.5rem; }
  .section-title { margin-top: 0.75rem; font-size: 0.9em; color: #666; }
  .members { margin-left: 0.5rem; }
  .sig { white-space: pre-wrap; font-size: 0.95em; }
  .chip { display:inline-block; padding: 0 6px; border-radius: 10px; font-size: 0.75em; background: #eef; color: #335; margin-left: 6px; }
  .name { font-weight: 600; }
  .mono-muted { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; color:#555; }
  .asm-badge { margin-left: 8px; }
  :target { background: #fffbe6; transition: background 0.8s ease; scroll-margin-top: 80px; }
  .grid { display: grid; grid-template-columns: 1fr; gap: 0.25rem; }
  ul { margin: 0.25rem 0 0.25rem 1.2rem; }
  :root{
    --bg:#fff; --fg:#111; --muted:#666; --border:#e5e5e5; --card:#fafafa;
    --pill-bg:#eee; --pill-fg:#333; --chip-bg:#eef; --chip-fg:#335;
    --target:#fffbe6; --link:#0b57d0; --shadow:rgba(0,0,0,.06);
  }
  html.dark{
    --bg:#0f1115; --fg:#e6e6e6; --muted:#a0a0a0; --border:#2a2f3a; --card:#151922;
    --pill-bg:#1f2430; --pill-fg:#cbd5e1; --chip-bg:#1b2a4a; --chip-fg:#c9defa;
    --target:#1d2330; --link:#8ab4f8; --shadow:rgba(0,0,0,.4);
  }

  /* swap colors in existing rules */
  html { scroll-behavior: smooth; }
  body { background:var(--bg); color:var(--fg); }
  a { color:var(--link); }
  .masthead { background:var(--bg); border-bottom:1px solid var(--border); }
  .masthead.is-stuck { box-shadow: 0 2px 10px var(--shadow); }
  .muted, .mono-muted { color:var(--muted); }
  details { border:1px solid var(--border); background:var(--card); }
  summary { color:var(--fg); }
  .pill { background:var(--pill-bg); color:var(--pill-fg); }
  .chip { background:var(--chip-bg); color:var(--chip-fg); }
  :target { background:var(--target); }
</style>
</head>
<body>
<div class="masthead">
  <div class="asm-header">
    <h1>.NET API Browser</h1>
    <div class="muted small">
      <span class="mono-muted">Assemblies:</span>""")

    # List assembly names
    names = sorted(set(a.GetName().Name for a in assemblies))
    parts.append(" " + ", ".join(f"<span class='pill'>{h(n)}</span>" for n in names))
    parts.append("""</div>
  </div>
  <div class="toolbar">
    <button id="expand-all" type="button">Expand all</button>
    <button id="collapse-all" type="button">Collapse all</button>
    <button id="theme-toggle" type="button" aria-pressed="false">🌙 Dark</button>
    <div class="search">
      <input id="filter" type="search" placeholder="Filter types (regex or text)…" />
    </div>
  </div>
</div>
""")

    # First pass: create docid anchors for members (we’ll also need them for <see cref=...>)
    for asm, t in types:
        asmname = asm.GetName().Name
        # Constructors
        for c in t.GetConstructors(bf):
            for did in id_candidates(id_ctor, c):
                docid_to_anchor[did] = anchor_from_docid(did, asmname)
        # Methods
        for m in [m for m in t.GetMethods(bf) if not m.IsSpecialName]:
            for did in id_candidates(id_method, m):
                docid_to_anchor[did] = anchor_from_docid(did, asmname)
        # Properties
        for p in t.GetProperties(bf):
            for did in id_candidates(id_property, p):
                docid_to_anchor[did] = anchor_from_docid(did, asmname)
        # Fields
        for f in [f for f in t.GetFields(bf) if not f.IsSpecialName]:
            for did in id_candidates(id_field, f):
                docid_to_anchor[did] = anchor_from_docid(did, asmname)
        # Events
        for e in t.GetEvents(bf):
            for did in id_candidates(id_event, e):
                docid_to_anchor[did] = anchor_from_docid(did, asmname)

    # Render each type
    for asm, t in types:
        asmname = asm.GetName().Name
        tkind = type_kind(t)
        tname = (t.FullName or t.Name).replace("+",".")
        tanchor = type_anchor[(asm.FullName, t)]
        vis = "public" if (t.IsPublic or t.IsNestedPublic) else (
              "protected" if t.IsNestedFamily else (
              "internal" if t.IsNestedAssembly else (
              "protected-internal" if getattr(t, "IsNestedFamORAssem", False) else (
              "private-protected" if getattr(t, "IsNestedFamANDAssem", False) else "private"))))

        spanClassPill = '<span class="pill">'


        parts.append(f'<details class="type" data-name="{h(tname)}" open="open">')
        parts.append(f'<summary id="{h(tanchor)}"><span class="kind">{h(tkind)}</span> '
                     f'<span class="name">{h(tname)}</span>'
                     f'<span class="pill">{h(vis)}</span>'
                     f'<span class="pill asm-badge">{h(asmname)}</span>')
        if getattr(t, "IsAbstract", False):
            parts.append('<span class="chip">abstract</span>')
        if getattr(t, "IsSealed", False):
            parts.append('<span class="chip">sealed</span>')
        parts.append('</summary>')

        # Docs for the type
        docmap = doc_maps.get(asm.FullName, {})
        tdoc = None
        for cand in id_candidates(id_type, t):
            if cand in docmap:
                tdoc = docmap[cand]; break

        base = t.BaseType
        if base is not None and not t.IsEnum:
            parts.append('<div class="section-title">Base type</div>')
            parts.append(f'<div class="mono">{link_type(base)}</div>')

        ifaces = sorted(list(t.GetInterfaces()), key=lambda i: friendly_type(i))
        if ifaces:
            parts.append('<div class="section-title">Implements</div><ul>')
            for i in ifaces:
                parts.append(f'<li class="mono">{link_type(i)}</li>')
            parts.append('</ul>')

        if tdoc is not None:
            parts.append('<div class="section-title">Documentation</div>')
            parts.append(render_doc(tdoc, docid_to_anchor))

        # Enum values
        if t.IsEnum:
            fields = [f for f in t.GetFields(bf) if f.IsLiteral]
            if fields:
                parts.append('<div class="section-title">Enum values</div><ul>')
                for f in sorted(fields, key=lambda f: f.Name):
                    val = ""
                    try: val = str(f.GetRawConstantValue())
                    except: pass
                    parts.append("<li><span class='name'>{}</span>{}</li>".format(
                        h(f.Name),
                        f" <span class='mono muted'>= {h(val)}</span>" if val else ""))
                parts.append('</ul>')

        parts.append('<div class="members">')

        # Constructors
        ctors = t.GetConstructors(bf)
        if ctors:
            parts.append('<div class="section-title">Constructors</div><div class="grid">')
            for c in sorted(ctors, key=lambda c: (vis_from_methodbase(c), c.GetParameters().Length)):
                did = id_candidates(id_ctor, c)[0]  # '.' first
                anchor = docid_to_anchor.get(did) or docid_to_anchor.get(id_candidates(id_ctor, c)[1])
                ps = []
                for p in c.GetParameters():
                    mod = "out " if (p.IsOut and not p.ParameterType.IsByRef) else ("ref " if p.ParameterType.IsByRef else "")
                    ps.append(f"{mod}{link_type(p.ParameterType)} {h(p.Name)}")
                parts.append('<div>')
                parts.append(f'<div class="sig mono" id="{h(anchor)}">(' + join_comma(ps) + ')'
                             f' {spanClassPill}{h(vis_from_methodbase(c))}</span>'
                             f'{ " "+spanClassPill+"static</span>" if c.IsStatic else ""}</div>')
                # docs
                cdoc = None
                for cand in id_candidates(id_ctor, c):
                    if cand in docmap: cdoc = docmap[cand]; break
                parts.append(render_doc(cdoc, docid_to_anchor))
                parts.append('</div>')
            parts.append('</div>')

        # Methods
        methods = [m for m in t.GetMethods(bf) if not m.IsSpecialName]
        if methods:
            parts.append('<div class="section-title">Methods</div><div class="grid">')
            for m in sorted(methods, key=lambda m: (m.Name, len(m.GetParameters()))):
                dids = id_candidates(id_method, m)
                anchor = docid_to_anchor.get(dids[0]) or docid_to_anchor.get(dids[1])
                ps = []
                for p in m.GetParameters():
                    mod = "out " if (p.IsOut and not p.ParameterType.IsByRef) else ("ref " if p.ParameterType.IsByRef else "")
                    ps.append(f"{mod}{link_type(p.ParameterType)} {h(p.Name)}")
                ret = link_type(m.ReturnType)
                garity = ""
                try:
                    if m.IsGenericMethodDefinition or m.IsGenericMethod:
                        garity = f"``{len(m.GetGenericArguments())}"
                except: pass
                parts.append('<div>')
                parts.append(f'<div class="sig mono" id="{h(anchor)}">{ret} {h(m.Name)}{h(garity)}(' + join_comma(ps) + ')'
                             f' <span class="pill">{h(vis_from_methodbase(m))}</span>'
                             f'{ " " + spanClassPill + "static</span>" if m.IsStatic else ""}'
                             f'{ " " + spanClassPill + "abstract</span>" if m.IsAbstract else ""}'
                             f'{ " " + spanClassPill + "virtual</span>" if (m.IsVirtual and not m.IsFinal) else ""}'
                             f'</div>')
                mdoc = None
                for cand in dids:
                    if cand in docmap: mdoc = docmap[cand]; break
                parts.append(render_doc(mdoc, docid_to_anchor))
                parts.append('</div>')
            parts.append('</div>')

        # Properties
        props = t.GetProperties(bf)
        if props:
            parts.append('<div class="section-title">Properties</div><div class="grid">')
            for p in sorted(props, key=lambda p: p.Name):
                dids = id_candidates(id_property, p)
                anchor = docid_to_anchor.get(dids[0]) or docid_to_anchor.get(dids[1])
                idx_params = p.GetIndexParameters()
                idx = ""
                if idx_params and idx_params.Length > 0:
                    items = []
                    for ip in idx_params:
                        mod = "out " if (ip.IsOut and not ip.ParameterType.IsByRef) else ("ref " if ip.ParameterType.IsByRef else "")
                        items.append(f"{mod}{link_type(ip.ParameterType)} {h(ip.Name)}")
                    idx = " <span class='small muted'>Indexer: (" + join_comma(items) + ")</span>"
                g = p.GetGetMethod(True)
                s = p.GetSetMethod(True)
                parts.append('<div>')
                parts.append(f'<div class="sig mono" id="{h(anchor)}">{link_type(p.PropertyType)} {h(p.Name)}'
                             f' {{ {"get;" if g else ""} {"set;" if s else ""} }}'
                             f' <span class="pill">{h(vis_from_property(p))}</span></div>')
                parts.append(idx)
                pdoc = None
                for cand in dids:
                    if cand in docmap: pdoc = docmap[cand]; break
                parts.append(render_doc(pdoc, docid_to_anchor))
                parts.append('</div>')
            parts.append('</div>')

        # Fields
        fields = [f for f in t.GetFields(bf) if not f.IsSpecialName]
        if fields:
            parts.append('<div class="section-title">Fields</div><div class="grid">')
            for f in sorted(fields, key=lambda f: f.Name):
                dids = id_candidates(id_field, f)
                anchor = docid_to_anchor.get(dids[0]) or docid_to_anchor.get(dids[1])
                value_str = ""
                if f.IsLiteral:
                    try:
                        value_str = str(f.GetRawConstantValue())
                    except:
                        pass
                parts.append('<div>')
                parts.append(f'<div class="sig mono" id="{h(anchor)}">{link_type(f.FieldType)} {h(f.Name)}'
                             f'{ " " + spanClassPill + "const</span>" if f.IsLiteral else ""}'
                             f'{ " " + spanClassPill + "readonly</span>" if f.IsInitOnly else ""}'
                             f'{ " " + spanClassPill + "static</span>" if f.IsStatic else ""}'
                             f' <span class="pill">{h(vis_from_field(f))}</span>'
                             f'{ " " + spanClassPill + "= " + h(value_str) + "</span>" if value_str else ""}'
                             f'</div>')
                fdoc = None
                for cand in dids:
                    if cand in docmap: fdoc = docmap[cand]; break
                parts.append(render_doc(fdoc, docid_to_anchor))
                parts.append('</div>')
            parts.append('</div>')

        # Events
        events = t.GetEvents(bf)
        if events:
            parts.append('<div class="section-title">Events</div><div class="grid">')
            for e in sorted(events, key=lambda e: e.Name):
                dids = id_candidates(id_event, e)
                anchor = docid_to_anchor.get(dids[0]) or docid_to_anchor.get(dids[1])
                addm = e.GetAddMethod(True)
                parts.append('<div>')
                parts.append(f'<div class="sig mono" id="{h(anchor)}">{link_type(e.EventHandlerType)} {h(e.Name)}'
                             f' <span class="pill">{h(vis_from_methodbase(addm) if addm else "public")}</span>'
                             f'{ " " + spanClassPill + "static</span>" if (addm and addm.IsStatic) else ""}'
                             f'</div>')
                edoc = None
                for cand in dids:
                    if cand in docmap: edoc = docmap[cand]; break
                parts.append(render_doc(edoc, docid_to_anchor))
                parts.append('</div>')
            parts.append('</div>')

        # Nested types (names only)
        nested = t.GetNestedTypes(bf)
        if nested and len(nested) > 0:
            parts.append('<div class="section-title">Nested types</div><ul>')
            for nt in sorted(nested, key=lambda x: x.Name):
                # link if we have it in the map
                key = (nt.Assembly.FullName, nt)
                label = (nt.FullName or nt.Name).replace("+",".")
                if key in type_anchor:
                    parts.append(f'<li><span class="kind">{h(type_kind(nt))}</span> '
                                 f'<a class="mono" href="#{h(type_anchor[key])}">{h(label)}</a></li>')
                else:
                    parts.append(f'<li><span class="kind">{h(type_kind(nt))}</span> <span class="mono">{h(label)}</span></li>')
            parts.append('</ul>')

        parts.append('</div>')  # .members
        parts.append('</details>')

    # Footer scripts (expand/collapse, filter, sticky shadow)
    parts.append("""
<script>
document.addEventListener('DOMContentLoaded', function () {
  const expandAll = () => document.querySelectorAll('details').forEach(d => d.open = true);
  const collapseAll = () => document.querySelectorAll('details').forEach(d => d.open = false);
  const ex = document.getElementById('expand-all');
  const co = document.getElementById('collapse-all');
  if (ex) ex.addEventListener('click', expandAll);
  if (co) co.addEventListener('click', collapseAll);
  document.addEventListener('keydown', (e) => {
    if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
    if (e.key.toLowerCase() === 'e') expandAll();
    if (e.key.toLowerCase() === 'c') collapseAll();
  });

  // Simple filter (regex or text)
  const filter = document.getElementById('filter');
  if (filter) {
    const types = Array.from(document.querySelectorAll('details.type'));
    filter.addEventListener('input', () => {
      const q = filter.value.trim();
      let rx = null;
      if (q.length > 0) {
        try { rx = new RegExp(q, 'i'); } catch(e) { rx = new RegExp(q.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'), 'i'); }
      }
      types.forEach(d => {
        const name = d.getAttribute('data-name') || '';
        d.style.display = (!rx || rx.test(name)) ? '' : 'none';
      });
    });
  }

  // Sticky masthead shadow toggle
  const mh = document.querySelector('.masthead');
  const onScroll = () => { if (mh) mh.classList.toggle('is-stuck', window.scrollY > 0); };
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
});

// Theme toggle with OS preference + persistence
const root = document.documentElement;
const btnTheme = document.getElementById('theme-toggle');

function setTheme(mode){ // mode: 'dark' | 'light'
  root.classList.toggle('dark', mode === 'dark');
  if (btnTheme){
    const dark = root.classList.contains('dark');
    btnTheme.textContent = dark ? '☀️ Light' : '🌙 Dark';
    btnTheme.setAttribute('aria-pressed', dark ? 'true' : 'false');
  }
}

// initialize from localStorage or OS
(function initTheme(){
  const stored = localStorage.getItem('theme'); // 'dark' | 'light' | null
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  setTheme(stored ?? (prefersDark ? 'dark' : 'light'));
  // keep following OS changes if user hasn't made a choice
  if (window.matchMedia){
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener?.('change', (e)=>{
      if (!localStorage.getItem('theme')) setTheme(e.matches ? 'dark' : 'light');
    });
  }
})();

// click handler
btnTheme?.addEventListener('click', ()=>{
  const next = root.classList.contains('dark') ? 'light' : 'dark';
  localStorage.setItem('theme', next);
  setTheme(next);
});
</script>
</body>
</html>""")

    html_out = "".join(parts)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_out)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
