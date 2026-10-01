#! python3
# -*- coding: utf-8 -*-
"""
LIVE EXHIBITION UPDATE  -  Exhibition Layout V1.3dm
IARC 425 . Reading Big Data at the Human Scale

Run inside Rhino 8:   RunPythonScript  ->  pick this file
(or open it in ScriptEditor and press Run). Re-run any time to refresh the
room from the live APIs - everything this script made last time is deleted
and rebuilt, so it never piles up.

The existing points on the layers  fans / heat pump / humidifier / lights
are NOT moved or deleted. They are the placement controls: move a point,
re-run, and the unit follows it.

    System        API                         Drives
    ------------  --------------------------  -------------------------------
    4 fans        24cocktails.com  (abv)      fan speed  (higher ABV = faster)
    4 heat pumps  24cocktails.com  (method)   discharge temperature
    2 humidifiers Open-Meteo  (30-day RH)     humidifier output  (%RH)
    6 lights      Open Brewery DB  (by state) light intensity 0..1

Every generated object carries Rhino user text (Properties > Attribute User
Text) with its region, source API, raw value and setpoint, so Grasshopper
can read it back with the  Get User Text  component.
"""

import json
import math
from datetime import datetime

import System
from System.Collections.Generic import List
import Rhino
import Rhino.Geometry as rg
import scriptcontext as sc

doc = sc.doc

# =============================================================================
# SETTINGS
# =============================================================================
TAG_KEY = "live_api"          # every generated object gets live_api = 1
LIVE_ROOT = "LIVE_API"        # parent layer for everything generated
ROOM_H = 4.0                  # m, ceiling height (matches app.py)
FAN_Z = 1.2                   # m, fan hub height (the sim's slice height)
FAN_MAX_MS = 6.0              # m/s at full speed (matches app.py)
FAN_FULL_ABV = 25.0           # % ABV that means full speed

# --- 1. Fans: regional cocktails ---------------------------------------------
# Fans are matched to regions by which corner of the room they sit in
# (Top view: +y is north, +x is east), not by object order.
FAN_REGIONS = {
    "NW": ("West Coast", ["Mai Tai", "Irish Coffee", "Pisco Punch",
                          "Lemon Drop", "Martinez", "Zombie"]),
    "NE": ("Northeast", ["Manhattan", "Cosmopolitan", "Bronx",
                         "Long Island Iced Tea", "Rob Roy", "Clover Club",
                         "Jack Rose", "Bloody Mary"]),
    "SE": ("South", ["Mint Julep", "Sazerac", "Hurricane", "Ramos Gin Fizz",
                     "Brandy Milk Punch", "Grasshopper", "Old Fashioned"]),
    "SW": ("Southwest", ["Margarita", "Paloma", "Ranch Water",
                         "Tequila Sunrise", "Mexican Mule"]),
}

# --- 2. Heat pumps: preparation method ---------------------------------------
# direction -> method. Temperatures are assigned by RANK of frequency from the
# live API, so if the ranking ever changes the temperatures follow the data.
HEATPUMP_DIRS = [("up", "shake", "Shaken", (0, 1)),
                 ("left", "stir", "Stirred", (-1, 0)),
                 ("right", "blend", "Blended", (1, 0)),
                 ("down", "muddle", "Muddled", (0, -1))]
RANK_TEMPS_F = [100.0, 80.0, 20.0, 10.0]   # most common -> least common
HEATPUMP_OFFSET = 0.55                      # m from the heat-pump point

# --- 3. Humidifiers: spirit origins ------------------------------------------
HUMIDIFIER_SITES = [  # sorted left -> right
    ("Tequila", "Jalisco, Mexico (Guadalajara)", 20.6597, -103.3496),
    ("Whiskey", "Dublin, Ireland", 53.3498, -6.2603),
]

# --- 4. Lights: brewery regions ----------------------------------------------
# Grid of the six lights in Top view:   NW  N-C  NE   (upper row)
#                                       SW  S-C  SE   (lower row)
BREWERY_REGIONS = {
    "NW": ("Northwest", ["washington", "oregon", "idaho", "montana",
                         "wyoming", "alaska"]),
    "NC": ("North Central / Midwest", ["north dakota", "south dakota",
                                       "nebraska", "kansas", "minnesota",
                                       "iowa", "missouri", "wisconsin",
                                       "illinois", "michigan", "indiana",
                                       "ohio"]),
    "NE": ("Northeast", ["maine", "new hampshire", "vermont", "massachusetts",
                         "rhode island", "connecticut", "new york",
                         "new jersey", "pennsylvania"]),
    "SW": ("Southwest / Pacific", ["california", "nevada", "utah", "colorado",
                                   "arizona", "new mexico", "hawaii"]),
    "SC": ("South Central", ["texas", "oklahoma", "arkansas", "louisiana"]),
    "SE": ("Southeast", ["delaware", "maryland", "district of columbia",
                         "virginia", "west virginia", "kentucky", "tennessee",
                         "north carolina", "south carolina", "georgia",
                         "florida", "alabama", "mississippi"]),
}
LIGHT_MIN = 0.10              # dimmest a light is allowed to go

# Snapshot from 2026-09-29, used ONLY if an API cannot be reached.
FALLBACK = {
    "abv": {"NW": 17.11, "NE": 19.49, "SE": 21.32, "SW": 14.93},
    "methods": {"shake": 4229, "stir": 1441, "blend": 164, "muddle": 94},
    "humidity": [78.8, 73.3],
    "breweries": {"NW": 1095, "NC": 1918, "NE": 1472,
                  "SW": 1706, "SC": 485, "SE": 1548},
}

# =============================================================================
# HTTP
# =============================================================================
try:
    System.Net.ServicePointManager.SecurityProtocol = \
        System.Net.SecurityProtocolType.Tls12
except Exception:
    pass


def get_json(url):
    wc = System.Net.WebClient()
    wc.Headers.Add("User-Agent", "IARC425-ExhibitionTemplate/1.0 (Rhino)")
    wc.Encoding = System.Text.Encoding.UTF8
    try:
        return json.loads(wc.DownloadString(url))
    finally:
        wc.Dispose()


def url_q(s):
    return System.Uri.EscapeDataString(s)


COCKTAIL_API = "https://24cocktails.com/api/v1/search"


def fetch_region_abv(names):
    """Average ABV of every 24cocktails recipe whose name contains one of the
    region's signature cocktails (variants included, abv=0 = unknown, skipped)."""
    total, n = 0.0, 0
    for name in names:
        data = get_json("%s?limit=50&q=%s" % (COCKTAIL_API, url_q(name)))
        for r in data.get("results", []):
            abv = r.get("abv") or 0
            if abv > 0 and name.lower() in (r.get("name_en") or "").lower():
                total += abv
                n += 1
    return (total / n if n else 0.0), n


def fetch_method_counts():
    out = {}
    for _, key, _, _ in HEATPUMP_DIRS:
        data = get_json("%s?limit=1&method=%s" % (COCKTAIL_API, key))
        out[key] = int(data.get("total", 0))
    return out


def fetch_humidity():
    lats = ",".join(str(s[2]) for s in HUMIDIFIER_SITES)
    lons = ",".join(str(s[3]) for s in HUMIDIFIER_SITES)
    url = ("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s"
           "&daily=relative_humidity_2m_mean&past_days=30&forecast_days=0"
           "&timezone=auto" % (lats, lons))
    data = get_json(url)
    if isinstance(data, dict):
        data = [data]
    means = []
    for loc in data:
        vals = [v for v in loc["daily"]["relative_humidity_2m_mean"]
                if v is not None]
        means.append(float(sum(vals)) / len(vals))
    return means


def fetch_breweries():
    data = get_json("https://api.openbrewerydb.org/v1/breweries/meta"
                    "?by_country=united_states")
    by_state = {}
    for k, v in data.get("by_state", {}).items():
        by_state[k.lower()] = by_state.get(k.lower(), 0) + int(v)
    return dict((code, sum(by_state.get(s, 0) for s in states))
                for code, (_, states) in BREWERY_REGIONS.items())


def live_or_fallback(label, fn, fallback):
    try:
        v = fn()
        print("  [live]     %s" % label)
        return v, "live"
    except Exception as e:
        print("  [FALLBACK] %s  (%s)" % (label, e))
        return fallback, "cached 2026-09-29"


# =============================================================================
# DOC HELPERS
# =============================================================================
def layer_index(full_path, color=None, parent=None):
    idx = doc.Layers.FindByFullPath(full_path, -1)
    if idx >= 0:
        return idx
    lyr = Rhino.DocObjects.Layer()
    lyr.Name = full_path.split("::")[-1]
    if color is not None:
        lyr.Color = color
    if parent is not None:
        lyr.ParentLayerId = doc.Layers[layer_index(parent)].Id
    return doc.Layers.Add(lyr)


def points_on(layer_name):
    idx = doc.Layers.FindByFullPath(layer_name, -1)
    if idx < 0:
        raise Exception("Layer '%s' not found" % layer_name)
    pts = []
    for o in doc.Objects.FindByLayer(doc.Layers[idx]) or []:
        if isinstance(o.Geometry, rg.Point):
            pts.append(o.Geometry.Location)
    return pts


def attrs(layer, color, name, info):
    a = Rhino.DocObjects.ObjectAttributes()
    a.LayerIndex = layer
    a.ColorSource = Rhino.DocObjects.ObjectColorSource.ColorFromObject
    a.ObjectColor = color
    a.Name = name
    a.SetUserString(TAG_KEY, "1")
    for k, v in info.items():
        a.SetUserString(k, str(v))
    return a


def add(geom, a):
    if isinstance(geom, rg.Brep):
        return doc.Objects.AddBrep(geom, a)
    if isinstance(geom, rg.TextDot):
        return doc.Objects.AddTextDot(geom, a)
    if isinstance(geom, rg.Curve):
        return doc.Objects.AddCurve(geom, a)
    if isinstance(geom, rg.Line):
        return doc.Objects.AddLine(geom, a)
    raise Exception("unsupported geometry")


def arrow(start, end, a):
    a2 = a.Duplicate()
    a2.ObjectDecoration = Rhino.DocObjects.ObjectDecoration.EndArrowhead
    a2.PlotWeight = 0.5
    return doc.Objects.AddLine(rg.Line(start, end), a2)


def lerp_color(c0, c1, t):
    t = max(0.0, min(1.0, t))
    return System.Drawing.Color.FromArgb(
        int(c0[0] + (c1[0] - c0[0]) * t),
        int(c0[1] + (c1[1] - c0[1]) * t),
        int(c0[2] + (c1[2] - c0[2]) * t))


def clear_previous():
    n = 0
    for o in doc.Objects.FindByUserString(TAG_KEY, "1", True) or []:
        if doc.Objects.Delete(o, True):
            n += 1
    for i in range(doc.Lights.Count):
        lo = doc.Lights[i]
        if lo is not None and not lo.IsDeleted and \
                (lo.Attributes.Name or "").startswith("LIVE_"):
            doc.Lights.Delete(i, True)
            n += 1
    return n


def corner_code(p, c):
    return ("N" if p.Y >= c.Y else "S") + ("E" if p.X >= c.X else "W")


# =============================================================================
# BUILDERS
# =============================================================================
def build_fan(p, center, region, abv, n, speed, lyr, stamp, src):
    hub = rg.Point3d(p.X, p.Y, FAN_Z)
    target = rg.Point3d(center.X, center.Y, FAN_Z)
    d = target - hub
    d.Unitize()
    pl = rg.Plane(hub, d)
    t = speed / FAN_MAX_MS
    col = lerp_color((150, 190, 230), (10, 60, 170), t)
    info = {"system": "fan", "region": region, "api": "24cocktails.com",
            "avg_abv_pct": round(abv, 2), "recipes_averaged": n,
            "fan_speed_m_s": round(speed, 2), "power_0_1": round(t, 3),
            "facing": "exhibition centre", "data": src, "updated": stamp}
    a = attrs(lyr, col, "LIVE_fan_" + region, info)

    # housing ring, hub, blades
    back = rg.Plane(hub - d * 0.12, d)
    add(rg.Cylinder(rg.Circle(back, 0.45), 0.24).ToBrep(False, False), a)
    add(rg.Sphere(hub, 0.08).ToBrep(), a)
    blades = 5
    for k in range(blades):
        ang = 2 * math.pi * k / blades
        radial = pl.XAxis * math.cos(ang) + pl.YAxis * math.sin(ang)
        side = rg.Vector3d.CrossProduct(d, radial)
        root, tip = hub + radial * 0.09, hub + radial * 0.40
        pts = [root - side * 0.05, tip - side * 0.10,
               tip + side * 0.10, root + side * 0.05, root - side * 0.05]
        crv = rg.PolylineCurve(List[rg.Point3d](pts))
        # pitch: faster fans get a steeper blade angle
        crv.Transform(rg.Transform.Rotation(math.radians(15 + 30 * t),
                                            radial, hub))
        for b in rg.Brep.CreatePlanarBreps(crv, doc.ModelAbsoluteTolerance) or []:
            add(b, a)
    # air throw: arrow length proportional to speed
    arrow(hub + d * 0.3, hub + d * (0.3 + speed * 0.45), a)
    # stand
    add(rg.Line(rg.Point3d(p.X, p.Y, 0), hub + rg.Vector3d(0, 0, -0.45)), a)
    add(rg.TextDot("%s fan\n%.1f%% ABV avg\n%.1f m/s" % (region, abv, speed),
                   hub + rg.Vector3d(0, 0, 0.75)), a)


def build_heatpump(p, direction, label, vec, count, share, temp_f, rank,
                   lyr, stamp, src):
    dvec = rg.Vector3d(vec[0], vec[1], 0)
    perp = rg.Vector3d(-vec[1], vec[0], 0)
    base = p + dvec * HEATPUMP_OFFSET
    t = (temp_f - 10.0) / 90.0
    col = lerp_color((40, 110, 230), (225, 45, 30), t)
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    info = {"system": "heat_pump", "direction": direction, "method": label,
            "api": "24cocktails.com", "recipe_count": count,
            "share_pct": round(share * 100, 1), "rank": rank,
            "setpoint_F": round(temp_f, 1), "setpoint_C": round(temp_c, 1),
            "data": src, "updated": stamp}
    a = attrs(lyr, col, "LIVE_heatpump_" + direction, info)
    box = rg.Box(rg.Plane(base, perp, dvec), rg.Interval(-0.35, 0.35),
                 rg.Interval(-0.175, 0.175), rg.Interval(0, 0.9))
    add(box.ToBrep(), a)
    # grille on the discharge face
    face = base + dvec * 0.18
    for z in (0.3, 0.45, 0.6):
        add(rg.Line(face + perp * -0.28 + rg.Vector3d(0, 0, z),
                    face + perp * 0.28 + rg.Vector3d(0, 0, z)), a)
    # discharge arrow: longer = hotter
    start = face + rg.Vector3d(0, 0, 0.45)
    arrow(start, start + dvec * (0.25 + 1.2 * max(t, 0.05)), a)
    add(rg.TextDot("Heat pump %s - %s\n%d recipes (%.1f%%)\n%.0f F / %.0f C"
                   % (direction.upper(), label, count, share * 100,
                      temp_f, temp_c),
                   base + dvec * 0.9 + rg.Vector3d(0, 0, 1.1)), a)


def build_humidifier(p, spirit, place, rh, lyr, stamp, src):
    col = lerp_color((200, 225, 240), (20, 120, 200), (rh - 30) / 70.0)
    info = {"system": "humidifier", "spirit": spirit, "location": place,
            "api": "Open-Meteo", "rh_30day_mean_pct": round(rh, 1),
            "setpoint_RH_pct": round(rh, 1), "power_0_1": round(rh / 100, 3),
            "data": src, "updated": stamp}
    a = attrs(lyr, col, "LIVE_humidifier_" + spirit, info)
    base = rg.Plane(rg.Point3d(p.X, p.Y, 0), rg.Vector3d.ZAxis)
    add(rg.Cylinder(rg.Circle(base, 0.22), 0.7).ToBrep(True, True), a)
    # mist plume: radius and height scale with 30-day humidity
    r = 0.12 + 0.6 * rh / 100.0
    add(rg.Sphere(rg.Point3d(p.X, p.Y, 0.7 + r), r).ToBrep(), a)
    arrow(rg.Point3d(p.X, p.Y, 0.7), rg.Point3d(p.X, p.Y, 0.7 + 2 * r + 0.9 * rh / 100.0), a)
    add(rg.TextDot("%s humidifier\n%s\n30-day RH %.1f%%" % (spirit, place, rh),
                   rg.Point3d(p.X, p.Y, 0.7 + 2 * r + 1.2)), a)


def build_light(p, code, region, count, brightness, lyr, stamp, src):
    col = lerp_color((70, 60, 30), (255, 220, 90), brightness)
    info = {"system": "can_light", "region": region, "api": "Open Brewery DB",
            "brewery_count": count, "intensity_0_1": round(brightness, 3),
            "data": src, "updated": stamp}
    a = attrs(lyr, col, "LIVE_light_" + code, info)
    top = rg.Point3d(p.X, p.Y, ROOM_H)
    can = rg.Plane(rg.Point3d(p.X, p.Y, ROOM_H - 0.15), rg.Vector3d.ZAxis)
    add(rg.Cylinder(rg.Circle(can, 0.18), 0.15).ToBrep(True, True), a)
    # beam: floor pool radius proportional to brightness
    pool = 0.25 + 1.35 * brightness
    cone = rg.Cone(rg.Plane(rg.Point3d(p.X, p.Y, ROOM_H - 0.15),
                            -rg.Vector3d.ZAxis), ROOM_H - 0.15, pool)
    add(cone.ToBrep(False), a)
    disk = rg.Circle(rg.Plane(rg.Point3d(p.X, p.Y, 0.005), rg.Vector3d.ZAxis), pool)
    for b in rg.Brep.CreatePlanarBreps(disk.ToNurbsCurve(),
                                       doc.ModelAbsoluteTolerance) or []:
        add(b, a)
    add(rg.TextDot("%s\n%d breweries\nintensity %.2f" % (region, count, brightness),
                   rg.Point3d(p.X, p.Y, ROOM_H + 0.4)), a)
    # a real Rhino spotlight so renders respond too
    lt = rg.Light()
    lt.LightStyle = rg.LightStyle.WorldSpot
    lt.Location = rg.Point3d(p.X, p.Y, ROOM_H - 0.16)
    lt.Direction = rg.Vector3d(0, 0, -(ROOM_H - 0.16))
    lt.SpotAngleRadians = math.atan2(pool, ROOM_H - 0.15)
    lt.Intensity = brightness
    lt.Diffuse = System.Drawing.Color.FromArgb(255, 236, 200)
    lt.Name = "LIVE_light_" + code
    la = Rhino.DocObjects.ObjectAttributes()
    la.LayerIndex = lyr
    la.Name = "LIVE_light_" + code
    for k, v in info.items():
        la.SetUserString(k, str(v))
    doc.Lights.Add(lt, la)


# =============================================================================
# MAIN
# =============================================================================
def main():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("Live exhibition update  %s" % stamp)

    fan_pts = points_on("fans")
    hp_pts = points_on("heat pump")
    hum_pts = points_on("humidifier")
    light_pts = points_on("lights")
    if len(fan_pts) != 4 or len(hp_pts) < 1 or len(hum_pts) != 2 or len(light_pts) != 6:
        raise Exception("Expected 4 fan, 1 heat pump, 2 humidifier and 6 light "
                        "points; found %d / %d / %d / %d"
                        % (len(fan_pts), len(hp_pts), len(hum_pts), len(light_pts)))

    # exhibition centre = bounding-box centre of all placement points
    allp = fan_pts + hp_pts + hum_pts + light_pts
    cx = (min(q.X for q in allp) + max(q.X for q in allp)) / 2.0
    cy = (min(q.Y for q in allp) + max(q.Y for q in allp)) / 2.0
    center = rg.Point3d(cx, cy, 0)

    print("Fetching APIs ...")

    def _abv():
        return dict((c, fetch_region_abv(names))
                    for c, (_, names) in FAN_REGIONS.items())
    abv, abv_src = live_or_fallback("24cocktails ABV by region", _abv,
                                    dict((k, (v, 0)) for k, v in FALLBACK["abv"].items()))
    methods, m_src = live_or_fallback("24cocktails preparation methods",
                                      fetch_method_counts, FALLBACK["methods"])
    rh, rh_src = live_or_fallback("Open-Meteo 30-day humidity", fetch_humidity,
                                  FALLBACK["humidity"])
    brew, b_src = live_or_fallback("Open Brewery DB counts", fetch_breweries,
                                   FALLBACK["breweries"])

    removed = clear_previous()
    C = System.Drawing.Color
    L_FAN = layer_index(LIVE_ROOT + "::Fans", C.FromArgb(30, 90, 180), LIVE_ROOT)
    L_HP = layer_index(LIVE_ROOT + "::Heat Pumps", C.FromArgb(200, 60, 40), LIVE_ROOT)
    L_HUM = layer_index(LIVE_ROOT + "::Humidifiers", C.FromArgb(40, 140, 200), LIVE_ROOT)
    L_LT = layer_index(LIVE_ROOT + "::Lights", C.FromArgb(230, 190, 40), LIVE_ROOT)

    summary = {"updated": stamp, "fans": [], "heat_pumps": [],
               "humidifiers": [], "lights": []}

    # ---- 1. fans ------------------------------------------------------------
    for p in fan_pts:
        code = corner_code(p, center)
        region, _ = FAN_REGIONS[code]
        a, n = abv[code]
        speed = min(FAN_MAX_MS, FAN_MAX_MS * a / FAN_FULL_ABV)
        build_fan(p, center, region, a, n, speed, L_FAN, stamp, abv_src)
        summary["fans"].append({"region": region, "abv": round(a, 2),
                                "recipes": n, "speed_m_s": round(speed, 2),
                                "x": p.X, "y": p.Y})

    # ---- 2. heat pumps --------------------------------------------------------
    hp = hp_pts[0]
    ranked = sorted(methods, key=lambda k: -methods[k])
    total = float(sum(methods.values())) or 1.0
    for direction, key, label, vec in HEATPUMP_DIRS:
        rank = ranked.index(key)
        temp = RANK_TEMPS_F[rank]
        share = methods[key] / total
        build_heatpump(hp, direction, label, vec, methods[key], share, temp,
                       rank + 1, L_HP, stamp, m_src)
        summary["heat_pumps"].append({"direction": direction, "method": label,
                                      "count": methods[key],
                                      "share_pct": round(share * 100, 1),
                                      "setpoint_F": temp})

    # ---- 3. humidifiers -------------------------------------------------------
    for p, (spirit, place, _, _), value in zip(sorted(hum_pts, key=lambda q: q.X),
                                               HUMIDIFIER_SITES, rh):
        build_humidifier(p, spirit, place, value, L_HUM, stamp, rh_src)
        summary["humidifiers"].append({"spirit": spirit, "location": place,
                                       "rh_30day_pct": round(value, 1)})

    # ---- 4. lights ------------------------------------------------------------
    xs = sorted(set(round(q.X, 3) for q in light_pts))
    ymid = sum(q.Y for q in light_pts) / 6.0
    cols = ["W", "C", "E"]
    grid = {("N", "W"): "NW", ("N", "C"): "NC", ("N", "E"): "NE",
            ("S", "W"): "SW", ("S", "C"): "SC", ("S", "E"): "SE"}
    peak = float(max(brew.values())) or 1.0
    for p in light_pts:
        row = "N" if p.Y > ymid else "S"
        col = cols[min(2, xs.index(round(p.X, 3)))] if len(xs) == 3 else \
            ("W" if p.X < center.X - 0.5 else "E" if p.X > center.X + 0.5 else "C")
        code = grid[(row, col)]
        region = BREWERY_REGIONS[code][0]
        count = brew[code]
        brightness = max(LIGHT_MIN, count / peak)
        build_light(p, code, region, count, brightness, L_LT, stamp, b_src)
        summary["lights"].append({"region": region, "breweries": count,
                                  "intensity": round(brightness, 3)})

    doc.Views.Redraw()

    # write a JSON snapshot next to the .3dm for Grasshopper / app.py
    try:
        folder = System.IO.Path.GetDirectoryName(doc.Path) if doc.Path else None
        if folder:
            out = System.IO.Path.Combine(folder, "exports", "live_api_values.json")
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(out))
            System.IO.File.WriteAllText(out, json.dumps(summary, indent=2))
            print("Wrote %s" % out)
    except Exception as e:
        print("Could not write JSON snapshot: %s" % e)

    print("Removed %d old live objects." % removed)
    print(json.dumps(summary, indent=2))
    print("Done. Save the file (Ctrl+S) to keep this state.")


main()
