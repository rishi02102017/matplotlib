import traceback, datetime, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib import cbook
print("VERSION", mpl.__version__, np.__version__)

def check(name):
    def deco(f):
        try:
            r = f()
            print(f"[{name}] -> {r!r}")
        except Exception as e:
            print(f"[{name}] EXC {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
        plt.close("all")
        return f
    return deco

@check("eng_neg_rounding")
def _():
    f = mticker.EngFormatter(places=1)
    return [f.format_data(v) for v in (-999.97, 999.97, -0.0009999, 1e33, -1e33)]

@check("percent_neg_xmax")
def _():
    f = mticker.PercentFormatter(xmax=-1)
    fig, ax = plt.subplots(); ax.set_ylim(0, -1); ax.yaxis.set_major_formatter(f)
    return f(-0.25)

@check("logformatter_labelOnlyBase")
def _():
    f = mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False)
    fig, ax = plt.subplots(); ax.set_yscale("log"); ax.set_ylim(1, 1e5)
    ax.yaxis.set_major_formatter(f); fig.canvas.draw()
    return [f(v) for v in (1, 20, 100)]

@check("twoslope_inverse_masked")
def _():
    n = mcolors.TwoSlopeNorm(vcenter=0, vmin=-2, vmax=4)
    a = np.ma.masked_array([-2., 0, 4], mask=[0, 1, 0])
    return n.inverse(n(a)), n.inverse(0.25)

@check("boundarynorm_extend_inverse")
def _():
    n = mcolors.BoundaryNorm([0, 1, 2], ncolors=4, extend="both")
    return n([-1, 0.5, 1.5, 3])

@check("lognorm_inverse_scalar")
def _():
    n = mcolors.LogNorm(1, 100)
    return n.inverse(0.5), type(n.inverse(0.5))

@check("colormap_with_extremes_resampled")
def _():
    cm = mpl.colormaps["viridis"].with_extremes(bad="r", under="g", over="b")
    r = cm.resampled(5)
    rv = cm.reversed()
    return r.get_bad(), r.get_under(), rv.get_under(), rv.get_over()

@check("listed_reversed_extremes")
def _():
    cm = mcolors.ListedColormap(["r", "g", "b"]).with_extremes(under="k", over="w")
    rv = cm.reversed()
    return rv.get_under(), rv.get_over(), cm.resampled(2).get_under()

@check("to_rgba_tuple_alpha")
def _():
    return mcolors.to_rgba(("red", 0.5)), mcolors.to_rgba_array([("red", 0.5), ("blue", 0.2)])

@check("to_rgba_nan")
def _():
    return mcolors.to_rgba((np.nan, 0, 0))

@check("same_color_none")
def _():
    return mcolors.same_color("none", "none"), mcolors.same_color(["none", "r"], ["none", "red"])

@check("bar_dates_width_align_edge")
def _():
    fig, ax = plt.subplots()
    d = [datetime.date(2020, 1, i) for i in (1, 2, 3)]
    b = ax.bar(d, [1, 2, 3], width=datetime.timedelta(days=0.5), align="edge")
    return [p.get_x() for p in b]

@check("barh_categories_label")
def _():
    fig, ax = plt.subplots()
    b = ax.barh(["a", "b"], [1, 2], label=["x", "y"])
    return [p.get_label() for p in b]

@check("bar_label_list_container")
def _():
    fig, ax = plt.subplots()
    b = ax.bar([1, 2], [3, np.nan])
    return [t.get_text() for t in ax.bar_label(b)]

@check("hist_stacked_weights_range")
def _():
    fig, ax = plt.subplots()
    n, bins, p = ax.hist([[1, 2, 3], [2, 3, 4]], bins=3, range=(0, 6), weights=[[1, 1, 1], [2, 2, 2]], stacked=True)
    return n, bins

@check("hist_step_bottom_horizontal")
def _():
    fig, ax = plt.subplots()
    n, bins, p = ax.hist([1, 2, 2, 3], bins=3, bottom=10, histtype="step", orientation="horizontal")
    return p[0].get_xy()[:4]

@check("hist_empty_stacked")
def _():
    fig, ax = plt.subplots()
    return ax.hist([[], [1, 2]], stacked=True, bins=2)[0]

@check("pie_zero_total")
def _():
    fig, ax = plt.subplots()
    return ax.pie([0, 0])

@check("pie_normalize_false")
def _():
    fig, ax = plt.subplots()
    w, t, a = ax.pie([0.2, 0.3], normalize=False, autopct="%1.0f%%")
    return [x.get_text() for x in a]

@check("errorbar_masked")
def _():
    fig, ax = plt.subplots()
    eb = ax.errorbar([1, 2, 3], np.ma.masked_array([1, 2, 3], mask=[0, 1, 0]), yerr=[0.1, 0.1, 0.1])
    return eb.lines[2][0].get_segments()

@check("fill_between_where_interp")
def _():
    fig, ax = plt.subplots()
    x = np.array([0, 1, 2, 3.]); y = np.array([1, -1, 1, -1.])
    pc = ax.fill_between(x, y, 0, where=y > 0, interpolate=True)
    return [p.vertices.round(2).tolist() for p in pc.get_paths()]

@check("stairs_horizontal_baseline_arr_datalim")
def _():
    fig, ax = plt.subplots()
    ax.stairs([1, 2], [0, 1, 2], baseline=[-5, -3], orientation="horizontal")
    return ax.dataLim.bounds

@check("stem_bottom_horizontal")
def _():
    fig, ax = plt.subplots()
    c = ax.stem([1, 2], [3, 4], bottom=1, orientation="horizontal")
    return c.baseline.get_xydata(), [s.tolist() for s in c.stemlines.get_segments()]

@check("step_where_mid_dates")
def _():
    fig, ax = plt.subplots()
    l, = ax.step(np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"), [1, 2], where="mid")
    return l.get_path().vertices

@check("boxplot_whis_tuple_autorange")
def _():
    return cbook.boxplot_stats([1, 2, 3, 4, 100], whis=(0, 100), autorange=True)[0]["whishi"]

@check("boxplot_stats_bootstrap_labels")
def _():
    s = cbook.boxplot_stats(np.random.randn(20, 2), labels=["a", "b"], bootstrap=100)
    return [x["label"] for x in s]

@check("violin_quantiles_side")
def _():
    fig, ax = plt.subplots()
    v = ax.violinplot([np.arange(10.)], quantiles=[[0.25, 0.75]], side="low", showmeans=True)
    return v["cquantiles"].get_segments()

@check("scatter_masked_c")
def _():
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2, 3], [1, 2, 3], c=np.ma.masked_array([1, 2, 3], mask=[0, 1, 0]), s=[1, np.nan, 3])
    fig.canvas.draw()
    return sc.get_offsets()

@check("imshow_extent_origin")
def _():
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((2, 3)), origin="lower", extent=(0, 3, 2, 0))
    return im.get_extent(), ax.get_ylim()

@check("axline_log")
def _():
    fig, ax = plt.subplots()
    ax.set_xscale("log"); ax.set_yscale("log")
    l = ax.axline((1, 1), slope=1)
    fig.canvas.draw()
    return l.get_slope(), l.get_xy1()

@check("axline_set_xy2_none_slope")
def _():
    fig, ax = plt.subplots()
    l = ax.axline((0, 0), (1, 1))
    l.set_xy2(None); l.set_slope(2)
    fig.canvas.draw()
    return l.get_slope()

@check("annotate_offset_fontsize")
def _():
    fig, ax = plt.subplots()
    a = ax.annotate("x", (0.5, 0.5), xytext=(1, 1), textcoords="offset fontsize")
    fig.canvas.draw()
    return a.get_window_extent().bounds

@check("table_colwidths")
def _():
    fig, ax = plt.subplots()
    t = ax.table(cellText=[["a", "b"]], colWidths=[0.1, 0.2], rowLabels=["r"], colLabels=["c1", "c2"])
    fig.canvas.draw()
    return t[0, 0].get_width(), t[1, -1].get_text().get_text()

@check("dates_tz_formatter")
def _():
    import zoneinfo
    tz = zoneinfo.ZoneInfo("America/New_York")
    loc = mdates.AutoDateLocator(tz=tz)
    fmt = mdates.ConciseDateFormatter(loc, tz=tz)
    fig, ax = plt.subplots()
    d0 = datetime.datetime(2020, 3, 8, 0, tzinfo=datetime.timezone.utc)
    ax.plot([d0, d0 + datetime.timedelta(hours=10)], [0, 1])
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(fmt)
    fig.canvas.draw()
    return [t.get_text() for t in ax.get_xticklabels()], fmt.get_offset()

@check("date2num_naive_tz_rc")
def _():
    with mpl.rc_context({"timezone": "Asia/Kolkata"}):
        return mdates.num2date(0)

@check("rrulewrapper_tz")
def _():
    import zoneinfo
    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    r = mdates.rrulewrapper(mdates.DAILY, dtstart=datetime.datetime(2020, 3, 28, tzinfo=tz), count=3)
    return r.between(datetime.datetime(2020, 3, 27, tzinfo=tz), datetime.datetime(2020, 4, 1, tzinfo=tz), inc=True)

@check("rc_validate")
def _():
    out = []
    for k, v in [("lines.dash_capstyle", "round"), ("axes.titlelocation", "left"), ("legend.loc", "best"), ("hatch.linewidth", "1.5"), ("savefig.bbox", "None"), ("lines.markersize", "-1"), ("legend.framealpha", "2"), ("figure.figsize", "1, 2, 3")]:
        try:
            mpl.rcParams[k] = v; out.append((k, mpl.rcParams[k]))
        except Exception as e:
            out.append((k, "ERR", str(e)[:60]))
    mpl.rcdefaults()
    return out

@check("subfigures_ratios")
def _():
    fig = plt.figure()
    sf = fig.subfigures(2, 2, width_ratios=[1, 2], height_ratios=[2, 1], wspace=0.1)
    fig.canvas.draw()
    return [s.bbox.bounds for s in sf.flat]

@check("mosaic_per_subplot_kw")
def _():
    fig, axd = plt.subplot_mosaic("AB", per_subplot_kw={"A": {"projection": "polar"}, ("B",): {"facecolor": "r"}}, sharex=True)
    return axd["A"].name, axd["B"].get_facecolor()

@check("legend_empty_ncols")
def _():
    fig, ax = plt.subplots()
    ax.plot([1], label="a")
    lg = ax.legend(ncols=0)
    fig.canvas.draw()
    return lg

@check("savefig_metadata_png_none")
def _():
    import io
    fig, ax = plt.subplots()
    b = io.BytesIO()
    fig.savefig(b, format="png", metadata={"Software": None})
    return b.getvalue()[:4]

@check("savefig_svg_metadata_date")
def _():
    import io
    fig, ax = plt.subplots()
    b = io.StringIO()
    fig.savefig(b, format="svg", metadata={"Date": datetime.date(2020, 1, 1)})
    return "2020-01-01" in b.getvalue()

@check("transforms_bbox_union_empty")
def _():
    from matplotlib.transforms import Bbox
    return Bbox.union([])

@check("transforms_blended_inverted")
def _():
    from matplotlib import transforms as mt
    t = mt.blended_transform_factory(mt.Affine2D().scale(2), mt.Affine2D().scale(3))
    return t.inverted().transform([[2, 3]])

@check("normalize_fully_masked")
def _():
    n = mcolors.Normalize()
    r = n(np.ma.masked_all(3))
    return n.vmin, n.vmax, r

@check("powernorm_inverse")
def _():
    n = mcolors.PowerNorm(2, vmin=0, vmax=4)
    return n.inverse(n(np.array([-1., 1, 4])))

@check("symlognorm_inverse")
def _():
    n = mcolors.SymLogNorm(1, vmin=-10, vmax=10, base=10)
    return n.inverse(n([-10, -0.5, 0, 5]))

@check("funcnorm_inverse")
def _():
    n = mcolors.FuncNorm((np.sqrt, np.square), vmin=0, vmax=4)
    return n.inverse(n([1., 4]))

@check("centerednorm_halfrange")
def _():
    n = mcolors.CenteredNorm(vcenter=1)
    n.autoscale([0, 5])
    return n.vmin, n.vmax, n.halfrange

@check("multipleloc_offset")
def _():
    l = mticker.MultipleLocator(2, offset=0.5)
    return l.tick_values(0, 5)

@check("scalarformatter_offset_neg")
def _():
    fig, ax = plt.subplots()
    ax.set_ylim(-100005, -100001)
    fig.canvas.draw()
    return ax.yaxis.get_major_formatter().get_offset()

@check("strmethod_pos")
def _():
    return mticker.StrMethodFormatter("{x}-{pos}")(1, 2)

@check("funcformatter_offset")
def _():
    f = mticker.FuncFormatter(lambda x, p: str(x)); f.set_offset_string("o")
    return f.get_offset()

@check("logit_formatter")
def _():
    f = mticker.LogitFormatter(use_overline=True)
    fig, ax = plt.subplots(); ax.set_yscale("logit"); ax.set_ylim(0.01, 0.99)
    ax.yaxis.set_major_formatter(f); fig.canvas.draw()
    return [t.get_text() for t in ax.get_yticklabels()]
