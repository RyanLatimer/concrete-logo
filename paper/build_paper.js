// Builds paper/Latimer_concrete_LOGO.docx from text below + results/*.csv + figures/*.png.
// Run from repo root:  NODE_PATH=<dir with docx installed> node paper/build_paper.js
// (npm install docx into any directory, e.g. /tmp/docxbuild, and point NODE_PATH at its node_modules.)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType, Header,
  Footer, PageNumber, VerticalAlign, TabStopType, ExternalHyperlink,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const OUT = path.join(__dirname, "Latimer_concrete_LOGO.docx");

// ---------- typography ----------
const SERIF = "Palatino Linotype";
const SANS = "Arial";
const INK = "1F1F1F";
const INK2 = "55534F";
const MUTED = "8A8781";
const RULE = "3A3A3A";
const ACCENT = "1C5CAB";
const BODY = 21; // half-points (10.5 pt)

// ---------- data ----------
function readCsv(p) {
  const [head, ...rows] = fs.readFileSync(path.join(ROOT, p), "utf8").trim().split("\n");
  const cols = head.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [cols[i], v])));
}
const summary = readCsv("results/summary.csv");
const cv = readCsv("results/cv.csv");
const get = (model, split) => summary.find((r) => r.model === model && r.split === split);
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
const cvStats = (model) => {
  const r = cv.filter((x) => x.model === model);
  return { r2: mean(r.map((x) => +x.r2)), r2sd: sd(r.map((x) => +x.r2)), rmse: mean(r.map((x) => +x.rmse)) };
};

// ---------- inline markup: *italic*, ^sup^, ~sub~ ----------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*[^*]+\*|\^[^^]+\^|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    const inner = t.slice(1, -1);
    if (t[0] === "*") out.push(new TextRun({ text: inner, italics: true, ...base }));
    if (t[0] === "^") out.push(new TextRun({ text: inner, superScript: true, ...base }));
    if (t[0] === "~") out.push(new TextRun({ text: inner, subScript: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text), alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 110, line: 276 }, ...opts,
});
const PL = (text) => P(text, { alignment: AlignmentType.LEFT });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });

// ---------- figures ----------
function figure(file, widthIn, label, caption) {
  const buf = fs.readFileSync(path.join(ROOT, "figures", file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); // PNG IHDR
  const width = Math.round(widthIn * 96);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width, height: Math.round(width * h / w) },
        altText: { title: label, description: caption, name: label } })],
    }),
    caption_(label, caption),
  ];
}
function caption_(label, text, keepNext = false) {
  return new Paragraph({
    keepNext, alignment: AlignmentType.LEFT, spacing: { after: 200, line: 250 },
    indent: { left: 360, right: 360 },
    children: [new TextRun({ text: label + "  ", bold: true, font: SANS, size: 16, color: INK }),
      ...runs(text, { font: SANS, size: 16, color: INK2 })],
  });
}

// ---------- tables (three-rule style) ----------
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const rule = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: RULE });
function cell(children, width, { top, bottom, align = AlignmentType.LEFT, span = 1, shade } = {}) {
  return new TableCell({
    children: Array.isArray(children) ? children : [children],
    width: { size: width, type: WidthType.DXA }, columnSpan: span,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 50, bottom: 50, left: 70, right: 70 },
    shading: shade ? { fill: shade, type: ShadingType.CLEAR, color: "auto" } : undefined,
    borders: { top: top || NONE, bottom: bottom || NONE, left: NONE, right: NONE },
  });
}
const tpara = (text, { bold = false, align = AlignmentType.LEFT, size = 17, color = INK, font = SANS } = {}) =>
  new Paragraph({ alignment: align, keepNext: true, spacing: { after: 0, line: 240 }, children: runs(text, { bold, size, color, font }) });

// Table 1: split definitions
function table1() {
  const W = [950, 1900, 2150, 850, 800, 2710];
  const head = ["Split", "Train on", "Test on", "n train", "n test", "Question asked"];
  const rows = [
    ["Random", "random 70% of rows", "random 15% of rows", "721", "155", "Interpolation (control)"],
    ["A1", "age ≤ 28 d", "age ≥ 90 d", "749", "190", "Mature from young concrete"],
    ["A2", "age 3 or 7 d", "age 28 d", "260", "425", "28-day from early-age data"],
    ["A3 (×7)", "all other age bins", "one age bin", "605–971", "59–425", "Each age bin in turn"],
    ["C1", "no fly ash", "contains fly ash", "566", "464", "Unseen SCM (fly ash)"],
    ["C2", "no slag", "contains slag", "471", "559", "Unseen SCM (slag)"],
    ["C3", "no superplasticizer", "contains superplasticizer", "379", "651", "Unseen admixture"],
  ];
  const num = (i) => (i === 3 || i === 4 ? AlignmentType.RIGHT : AlignmentType.LEFT);
  return new Table({
    width: { size: W.reduce((a, b) => a + b), type: WidthType.DXA }, columnWidths: W,
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows: [
      new TableRow({ tableHeader: true, children: head.map((h, i) =>
        cell(tpara(h, { bold: true, align: num(i) }), W[i], { top: rule(12), bottom: rule(6) })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((v, i) =>
        cell(tpara(v, { bold: i === 0, align: num(i) }), W[i], { bottom: ri === rows.length - 1 ? rule(12) : undefined })) })),
    ],
  });
}

// Table 2: R² and RMSE for every model × main split
function fmtR2(v) {
  const a = Math.abs(v);
  const s = a >= 10 ? a.toFixed(0) : a.toFixed(2);
  return (v < 0 ? "−" : "") + s;
}
function table2() {
  const models = [["XGB", "XGBoost"], ["RF", "Random forest"], ["MLP", "Neural net (MLP)"], ["LR", "Linear regression"]];
  const splits = ["random", "cv", "A1", "A2", "C1", "C2", "C3"];
  const W = [1620, 1105, 1105, 1105, 1105, 1105, 1105, 1110];
  const val = (m, s) => {
    if (s === "cv") { const c = cvStats(m); return { r2: c.r2, sd: c.r2sd, rmse: c.rmse }; }
    const r = get(m, s); return { r2: +r.r2_mean, sd: +r.r2_std, rmse: +r.rmse_mean };
  };
  const best = Object.fromEntries(splits.map((s) => [s, Math.max(...models.map(([m]) => val(m, s).r2))]));
  const C = AlignmentType.CENTER;
  const group = new TableRow({ tableHeader: true, children: [
    cell(tpara(""), W[0], { top: rule(12) }),
    cell(tpara("Control", { bold: true, align: C, size: 16, color: INK2 }), W[1] + W[2], { top: rule(12), bottom: rule(4), span: 2 }),
    cell(tpara("Age held out", { bold: true, align: C, size: 16, color: INK2 }), W[3] + W[4], { top: rule(12), bottom: rule(4), span: 2 }),
    cell(tpara("Composition held out", { bold: true, align: C, size: 16, color: INK2 }), W[5] + W[6] + W[7], { top: rule(12), bottom: rule(4), span: 3 }),
  ] });
  const heads = ["Model", "Random", "5-fold CV", "A1", "A2", "C1", "C2", "C3"];
  const head = new TableRow({ tableHeader: true, children: heads.map((h, i) =>
    cell(tpara(h, { bold: true, align: i ? C : AlignmentType.LEFT }), W[i], { bottom: rule(6) })) });
  const body = models.map(([m, name], ri) => new TableRow({ cantSplit: true, children: [
    cell(tpara(name, { bold: true }), W[0], { bottom: ri === models.length - 1 ? rule(12) : undefined }),
    ...splits.map((s, i) => {
      const v = val(m, s);
      const isBest = v.r2 === best[s];
      const line1 = [new TextRun({ text: fmtR2(v.r2), bold: isBest, font: SANS, size: 17, color: INK })];
      if (v.sd >= 0.005) line1.push(new TextRun({ text: ` ±${v.sd >= 10 ? v.sd.toFixed(0) : v.sd.toFixed(2)}`, font: SANS, size: 13, color: INK2 }));
      return cell([
        new Paragraph({ alignment: C, keepNext: true, spacing: { after: 0 }, children: line1 }),
        new Paragraph({ alignment: C, keepNext: true, spacing: { after: 0 }, children: [new TextRun({ text: v.rmse.toFixed(1), font: SANS, size: 14, color: MUTED })] }),
      ], W[i + 1], { bottom: ri === models.length - 1 ? rule(12) : undefined, shade: ri % 2 === 0 ? "F6F5F2" : undefined });
    }),
  ] }));
  return new Table({
    width: { size: W.reduce((a, b) => a + b), type: WidthType.DXA }, columnWidths: W,
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows: [group, head, ...body],
  });
}

// ---------- references ----------
const REFS = [
  "Lehne, J. & Preston, F. *Making Concrete Change: Innovation in Low-carbon Cement and Concrete.* Chatham House Report (Royal Institute of International Affairs, London, 2018).",
  "Yeh, I.-C. Modeling of strength of high-performance concrete using artificial neural networks. *Cem. Concr. Res.* **28**, 1797–1808 (1998). https://doi.org/10.1016/S0008-8846(98)00165-3",
  "Yeh, I.-C. Concrete Compressive Strength [dataset]. UCI Machine Learning Repository (1998). https://doi.org/10.24432/C5PK67",
  "Feng, D.-C. et al. Machine learning-based compressive strength prediction for concrete: an adaptive boosting approach. *Constr. Build. Mater.* **230**, 117000 (2020). https://doi.org/10.1016/j.conbuildmat.2019.117000",
  "Zhang, W., Guo, J., Ning, C., Cheng, R. & Liu, Z. Prediction of concrete compressive strength using a Deepforest-based model. *Sci. Rep.* **14**, 18918 (2024). https://doi.org/10.1038/s41598-024-69616-9",
  "Fu, H., Zhou, X., Xu, P. & Sun, D. Prediction of compressive strength of concrete using explainable machine learning models. *Materials* **18**, 5009 (2025). https://doi.org/10.3390/ma18215009",
  "Li, Z. et al. Machine learning in concrete science: applications, challenges, and best practices. *npj Comput. Mater.* **8**, 127 (2022). https://doi.org/10.1038/s41524-022-00810-x",
  "Silva, V. P., Carvalho, R. A., Rêgo, J. H. S. & Evangelista, F. Machine learning-based prediction of the compressive strength of Brazilian concretes: a dual-dataset study. *Materials* **16**, 4977 (2023). https://doi.org/10.3390/ma16144977",
  "Babaei, H., Zamani, M. & Mohammadi, S. The impact of data splitting methods on machine learning models: a case study for predicting concrete workability. *Mach. Learn. Comput. Sci. Eng.* **1**, 21 (2025). https://doi.org/10.1007/s44379-025-00021-3",
  "Roberts, D. R. et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* **40**, 913–929 (2017). https://doi.org/10.1111/ecog.02881",
  "Li, Z. et al. Machine learning in concrete science: applications, challenges, and best practices [dataset]. Materials Data Facility (2022). https://doi.org/10.18126/8k1f-mx77",
  "Pedregosa, F. et al. Scikit-learn: machine learning in Python. *J. Mach. Learn. Res.* **12**, 2825–2830 (2011).",
  "Breiman, L. Random forests. *Mach. Learn.* **45**, 5–32 (2001). https://doi.org/10.1023/A:1010933404324",
  "Chen, T. & Guestrin, C. XGBoost: a scalable tree boosting system. In *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 785–794 (2016). https://doi.org/10.1145/2939672.2939785",
  "Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30*, 4765–4774 (2017).",
];
function refPara(text, i) {
  // **bold** for volume numbers, *italic*, URLs as links
  const parts = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|https?:\/\/\S+)/g;
  let last = 0, m;
  const base = { size: 17, font: SERIF, color: INK };
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) parts.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("*")) parts.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    else parts.push(new ExternalHyperlink({ link: t, children: [new TextRun({ text: t, ...base, color: ACCENT })] }));
    last = m.index + t.length;
  }
  if (last < text.length) parts.push(new TextRun({ text: text.slice(last), ...base }));
  return new Paragraph({
    spacing: { after: 70, line: 250 }, indent: { left: 400, hanging: 400 },
    tabStops: [{ type: TabStopType.LEFT, position: 400 }],
    children: [new TextRun({ text: `${i + 1}.\t`, ...base, color: INK2 }), ...parts],
  });
}

// ---------- content ----------
const R = (m, s) => get(m, s);
const f2 = (x) => (+x).toFixed(2);

const title = "Random Splits Overstate Concrete-Strength Prediction: A Leave-One-Group-Out Test Across Curing Age and Mix Composition";

const abstract = [
  "Machine-learning models trained on the public UCI concrete dataset routinely report R² above 0.90 for compressive strength. Those scores come from random train/test splits, in which the test rows closely resemble the training rows. We asked how the same models perform when the test set contains curing ages or mix ingredients that the training set lacks. On the 1,030-row dataset we trained linear regression, random forest, XGBoost and a small neural network with fixed settings, and scored each on a random-split control and twelve group-held-out splits over three seeds. XGBoost, the strongest model, fell from R² 0.92 (RMSE 4.7 MPa) on random splits to 0.39 (10.0 MPa) when trained on ages up to 28 days and tested on ages of 90 days and more, and to 0.67 (8.2 MPa) when trained on mixes without fly ash and tested on mixes with it. Predicting 28-day strength from 3- and 7-day data failed outright (R² 0.01). Linear and neural models extrapolated to physically impossible values. The age failures are systematic rather than noisy: tree models stop at the last strength level they saw and under-predict mature concrete by 7–10 MPa. Random splits also leak mix identity. The 1,030 rows describe only 426 distinct mixes, and 69–80% of random test rows share a mix with a training row. Random-split scores therefore measure interpolation among familiar mixes, not prediction for new ones, and should be reported alongside group-held-out results.",
];

const children = [];
const push = (...xs) => xs.flat().forEach((x) => children.push(x));

// Title block
push(
  new Paragraph({ spacing: { after: 160, line: 300 }, children: [new TextRun({ text: title, font: SANS, bold: true, size: 34, color: INK })] }),
  new Paragraph({ spacing: { after: 20 }, children: [new TextRun({ text: "Ryan Latimer", font: SANS, size: 21, bold: true, color: INK })] }),
  new Paragraph({ spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 10 } },
    children: [new TextRun({ text: "Harbor Springs High School, Harbor Springs, Michigan, USA", font: SANS, size: 18, color: INK2 })] }),
  new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "ABSTRACT", font: SANS, bold: true, size: 17, color: ACCENT, characterSpacing: 30 })] }),
  ...abstract.map((t) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 264 },
    children: runs(t, { size: 19 }) })),
  new Paragraph({ spacing: { after: 280 }, children: [
    new TextRun({ text: "Keywords  ", font: SANS, bold: true, size: 16, color: INK2 }),
    new TextRun({ text: "concrete compressive strength · machine learning · extrapolation · leave-one-group-out validation · supplementary cementitious materials", font: SANS, size: 16, color: INK2 })] }),
);

// 1 Introduction
push(
  H1("1  Introduction"),
  P("Compressive strength decides whether a batch of concrete is accepted, and it is normally measured by crushing test cylinders after 28 days of curing. A reliable way to estimate strength from the mix recipe would shorten mix design and reduce the habit of adding extra cement as a safety margin. That matters beyond cost, because cement production accounts for roughly 8% of global CO~2~ emissions [1]."),
  P("Machine learning is the usual tool for this estimate. Yeh [2] trained neural networks on 1,030 laboratory tests covering cement, slag, fly ash, water, superplasticizer, aggregates and curing age, and released the data through the UCI repository [3]. The set has since become a standard benchmark, and studies using boosted trees, deep forests and explainable ensembles routinely report R² above 0.90 on it and on similar laboratory data [4–6]."),
  P("Almost all of these scores come from random train/test splits. A random split asks whether a model can fill in gaps within data like the data it was trained on. The situations that motivate the work ask something harder: the strength of a mix at an age that has not been tested yet, or the strength of a mix containing a supplementary cementitious material (SCM) that was absent from the training data. Li et al. [7] identify data bias and the gap between laboratory data and real use as open problems for machine learning in concrete science. Silva et al. [8] showed the practical cost: models trained on Brazilian concretes transferred poorly to Yeh’s data and vice versa. Babaei et al. [9] found that the splitting method alone changes reported performance for concrete workability. In ecology, where data are often grouped by site or year, Roberts et al. [10] showed that random cross-validation on structured data overstates predictive skill, and recommended holding out whole groups."),
  P("On the UCI benchmark itself we found no study that holds out a curing age or a mix family while keeping models and features fixed. We ran that test. We held out groups defined by age (young versus mature concrete, and each age in turn) and by composition (mixes with and without fly ash, slag or superplasticizer), then compared each result with a random-split control. We expected tree ensembles to fall from about 0.90 to between 0.6 and 0.8. Composition shift fell roughly in that range, but age extrapolation was considerably worse, and the reasons for the failure are as informative as its size."),
);

// 2 Data
push(
  H1("2  Data"),
  P("We used the Foundry-ML repackaging of the UCI data, distributed through Hugging Face and linked to a Materials Data Facility deposit by Li et al. [11]. It contains 1,030 rows and seven inputs: cement, blast-furnace slag, fly ash, water, superplasticizer and coarse aggregate (all in kg/m³), and age at testing (days). Fine aggregate, the eighth input in Yeh’s original data [2], is absent from this version. We did not restore it, so our absolute scores are not directly comparable with studies that use all eight inputs. The comparisons between splits, which are the subject of this paper, are unaffected."),
  P("The target, compressive strength, ranges from 2.3 to 82.6 MPa (mean 35.8, SD 16.7). There are no missing values. Twenty-five rows are exact duplicates; we kept them and discuss their effect in Section 5.4. Age takes 14 distinct values from 1 to 365 days and is heavily concentrated at 28 days (425 rows, 41%). One further property shapes the results: the 1,030 rows describe only 426 distinct mixes, and 181 of those mixes were tested at more than one age."),
);

// 3 Methods
push(
  H1("3  Methods"),
  H2("3.1  Splits"),
  P("Table 1 lists every split. The random control draws 70% of rows for training and 15% for testing. The remaining 15% was set aside as a validation set but never used, because no model was tuned. A1 trains on ages up to 28 days and tests on ages of 90 days and more. The 56-day rows are excluded from both sides so that the test set sits well beyond the training range. A2 trains only on 3- and 7-day results and predicts 28-day strength, which is the prediction a producer would most want. A3 holds out one age bin at a time. The bins are 3 d (including the two 1-day rows), 7, 14, 28 and 56 d, 90 d (90–120 d) and 180+ d (180–365 d). The composition splits train on mixes that contain none of a given ingredient and test on mixes that contain it: fly ash (C1), slag (C2) or superplasticizer (C3)."),
  caption_("Table 1.", "Split definitions. Row counts are for the full 1,030-row dataset; A3 is seven separate splits, one per held-out age bin.", true),
  table1(),
  new Paragraph({ spacing: { after: 120 }, children: [] }),
  H2("3.2  Models"),
  P("We compared four models, each wrapped in a scikit-learn [12] pipeline that standardizes inputs using statistics from the training rows only. The models were linear regression; a random forest [13] with 200 trees; XGBoost [14] with 300 trees, maximum depth 6, learning rate 0.05 and row and column subsampling of 0.8; and a multilayer perceptron with hidden layers of 64 and 32 ReLU units, L2 penalty 0.001, and early stopping on 15% of the training rows. We fixed these settings in advance and used them unchanged on every split, so any difference between splits comes from the split alone. No features were derived from the target."),
  H2("3.3  Evaluation"),
  P("For every model and split we computed root-mean-square error (RMSE), mean absolute error (MAE) and R² on the test rows. MAE tracked RMSE closely and is given only in the repository’s result tables. R² is measured against the variance of the test set, so it can shift when the test set changes even if the size of the errors does not. We therefore read it together with RMSE. A negative R² means the model is worse than predicting the test-set mean. Each configuration was run with seeds 0, 1 and 2, and we report the mean and standard deviation. For the random split the seed changes both the model and the partition. For group splits the partition is fixed, so the spread reflects only model randomness, and linear regression has none. As a second control, we ran shuffled 5-fold cross-validation on the full dataset."),
  P("Three diagnostics used the same splits. (i) Error by age: a random forest trained on the A1 training rows was scored on every age bin. Bins inside the training range (≤28 d) were scored with 5-fold out-of-fold predictions so that every bin is evaluated on rows the model has not seen. (ii) Age response: for a random forest trained on the random-split training rows and one trained on the A1 training rows, we set every mix’s age to each value from 1 to 365 days and averaged the predictions. This is a partial-dependence curve. (iii) Attribution: SHAP values [15] for XGBoost and permutation importance for the random forest, both trained on the random split with seed 0."),
);

// 4 Results
const xr = R("XGB", "random"), fr = R("RF", "random");
const xa1 = R("XGB", "A1"), fa1 = R("RF", "A1");
push(
  H1("4  Results"),
  H2("4.1  Random versus group-held-out splits"),
  P(`Table 2 summarizes the main splits. The random split and 5-fold cross-validation agree with each other and with the literature: XGBoost reached R² ${f2(xr.r2_mean)} on the random test set and ${f2(cvStats("XGB").r2)} in cross-validation, and the random forest reached ${f2(fr.r2_mean)} and ${f2(cvStats("RF").r2)}. Every model scored lower on each fixed group split (A1, A2, C1–C3) than on its random split.`),
  P(`Age extrapolation produced the largest drop. On A1, XGBoost fell to R² ${f2(xa1.r2_mean)} and the random forest to ${f2(fa1.r2_mean)}, and RMSE roughly doubled, from ${(+xr.rmse_mean).toFixed(1)} to ${(+xa1.rmse_mean).toFixed(1)} MPa and from ${(+fr.rmse_mean).toFixed(1)} to ${(+fa1.rmse_mean).toFixed(1)} MPa. The doubling in RMSE shows that the drop is not an artefact of R²’s dependence on test-set variance. The gap is far larger than seed-to-seed variation. Even the weakest random-split seeds (R² 0.86 for the random forest and 0.90 for XGBoost) sit more than 0.5 above the A1 scores, whose seed-to-seed standard deviation was below 0.003. A2 failed for every model: the best, XGBoost, reached R² ${f2(R("XGB", "A2").r2_mean)} with an RMSE of ${(+R("XGB", "A2").rmse_mean).toFixed(1)} MPa, no better than predicting the mean 28-day strength. Linear regression and the neural network did far worse than failing. On A1 their RMSEs were ${(+R("LR", "A1").rmse_mean).toFixed(0)} and ${(+R("MLP", "A1").rmse_mean).toFixed(0)} MPa, larger than any strength in the dataset.`),
  caption_("Table 2.", "Test R² (large figure; ± seed SD shown when ≥ 0.005) and RMSE in MPa (small grey figure), mean of seeds 0–2. Bold marks the best model in each column. Split definitions are in Table 1.", true),
  table2(),
  new Paragraph({ spacing: { after: 120 }, children: [] }),
  P(`Composition shift was milder but still substantial. XGBoost scored ${f2(R("XGB", "C1").r2_mean)}, ${f2(R("XGB", "C2").r2_mean)} and ${f2(R("XGB", "C3").r2_mean)} on C1–C3, with RMSEs of ${(+R("XGB", "C1").rmse_mean).toFixed(1)} to ${(+R("XGB", "C3").rmse_mean).toFixed(1)} MPa. The random forest scored ${f2(R("RF", "C1").r2_mean)}, ${f2(R("RF", "C2").r2_mean)} and ${f2(R("RF", "C3").r2_mean)}. C3 is the only split on which linear regression beat both tree models (R² ${f2(R("LR", "C3").r2_mean)}, RMSE ${(+R("LR", "C3").rmse_mean).toFixed(1)} MPa).`),
  H2("4.2  Age errors are a systematic offset"),
  P("Figure 1 shows where the errors come from. On the random split, the random forest’s predictions scatter evenly around the diagonal (mean error −0.8 MPa). On A1, nearly every point lies below the diagonal, so mature concrete is predicted to be weaker than it is (mean error −9.3 MPa). On C1 the scatter is wider but roughly centred (+2.7 MPa)."),
  ...figure("fig1_pred_vs_true.png", 6.5, "Figure 1.", "Random-forest predictions against measured strength on three test sets (seed 0). Dashed line: perfect prediction. Bias is the mean of predicted minus measured strength."),
  P("Figure 2 breaks the A1 model’s error down by age. Inside the training range, out-of-fold RMSE is 4.4–6.2 MPa and the mean error stays within ±1.2 MPa. At 56, 90 and 180+ days RMSE rises to 8.0, 10.6 and 9.9 MPa, and the mean error falls to −7.2, −9.6 and −8.7 MPa. Almost all of the added error is this offset. Once the offset is removed, the remaining scatter at 90 days (√(10.6² − 9.6²) ≈ 4.4 MPa) matches the scatter within the training range. The model ranks mature mixes about as well as young ones. It simply places all of them about 9 MPa too low."),
  ...figure("fig2_error_vs_age.png", 4.9, "Figure 2.", "Error by age bin for a random forest trained on ages ≤ 28 d (A1 training set). Shaded bins are scored with 5-fold out-of-fold predictions; the unshaded bins were never seen in training. n is the number of rows per bin."),
  H2("4.3  Leave-one-age-bin-out"),
  P("Holding out one age bin at a time (Figure 3) gave an asymmetric result. Early bins were hard. With 3-day results held out, no model exceeded R² 0.06. Holding out 28 days (41% of the data) left R² between 0.36 and 0.53, even though 14- and 56-day data on either side remained in training. Late bins looked easy: XGBoost scored 0.74, 0.89 and 0.88 on the 56-, 90- and 180+-day bins, and the random forest scored 0.73, 0.84 and 0.75. The linear and neural models again failed at the edge of the range (R² −33 and −8.1 on 180+ days). The 14-day bin, with only 62 rows, split the two tree models (random forest −0.37, XGBoost 0.51 ± 0.14), and we do not read anything into that difference."),
  ...figure("fig3_a3_heatmap.png", 5.9, "Figure 3.", "Test R² when each age bin is held out in turn (mean of three seeds). Colour is centred on R² = 0, the score of always predicting the test-set mean, and clipped at −1. Printed values are unclipped."),
);

// 5 Discussion
push(
  H1("5  Discussion"),
  H2("5.1  Why age extrapolation fails"),
  P("A tree-based model predicts by averaging the training targets that fall in the same leaf. Beyond the largest age it has seen, every split on age sends a sample the same way, so the prediction stops changing. Figure 4 shows this directly. The random forest trained on all ages predicts a mean strength that climbs to 46 MPa by 56 days and reaches 48.7 MPa at one year. The same model trained on ages up to 28 days stays at 38.2 MPa at every age past 28. The 10 MPa gap between the curves matches the offset in Figure 2. Real concrete keeps gaining strength after 28 days as hydration continues, and slag and fly ash react especially slowly, but a model that has never seen that gain cannot predict it."),
  ...figure("fig4_age_response.png", 4.9, "Figure 4.", "Random-forest partial dependence on age: mean prediction over all 1,030 mixes as age is swept from 1 to 365 days, for a model trained on the random split (all ages) and one trained on the A1 training rows (≤ 28 d)."),
  P("Linear regression and the neural network do extrapolate, but nothing constrains them to follow the flattening strength–age curve, so their predictions run far outside physical limits. A2 fails by the same mechanism as A1. A model that has seen only 3- and 7-day strengths has no information about the gain between 7 and 28 days."),
  H2("5.2  Why late held-out ages look easy"),
  P("The A3 asymmetry follows the shape of the strength–age curve: steep before 28 days, nearly flat after about 56 (Figure 4, blue curve). When the 180+ bin is held out, a tree model predicts from its 90–120-day leaves, and because strength barely changes over that interval, the frozen prediction is close to correct. When the 3- or 14-day bin is held out, the nearest ages in training differ in strength by several MPa, and the frozen prediction is wrong. Whether age extrapolation works therefore depends less on its direction than on how much strength changes across the gap. A benchmark that held out only late ages would wrongly conclude that the problem is solved."),
  H2("5.3  Composition shift"),
  P("In C1 every training mix has zero fly ash, so the fly-ash input is constant and no model can learn its effect. The fly-ash mixes also differ in other ways. They average 241 kg/m³ of cement against 314 kg/m³ for the rest, and about twice as much superplasticizer (8.8 against 4.1 kg/m³), so the models are also predicting in parts of the other inputs that they have seldom seen. This explains an apparent contradiction. In the random-split model fly ash has the smallest SHAP attribution (mean |SHAP| 0.5 MPa, against 7.7 for age and 7.0 for cement; permutation importance gives the same order at the top), yet holding out fly-ash mixes costs 0.25–0.32 in R². Feature importance measured on a random split describes how much an input moves predictions among familiar mixes. It does not describe how much the model will lose when a whole family of mixes is new. The superplasticizer-free mixes in C3 are a distinct family (196 against 173 kg/m³ of water, and a mean strength of 29.5 against 39.5 MPa). Linear regression’s lead on C3 is consistent with a roughly monotonic water and admixture effect, which a straight line extends better than a tree does."),
  H2("5.4  Why random splits flatter"),
  P("The dataset has 1,030 rows but only 426 distinct mixes. Across the three random splits, 69–80% of test rows share their mix with a training row tested at a different age, and 1–3% have an exact duplicate in training. For those rows the model only has to place the test point on a strength curve it has already partly seen. That is interpolation within a known recipe, not prediction for a new one, and it explains much of the gap between the random-split scores and the composition splits. Grouping the split by mix, so that every age of a recipe falls on the same side, would be a natural further control; we did not run it."),
  H2("5.5  Recommendations"),
  P("Random-split scores on this dataset should be read as an upper bound. A study that claims a model is useful should also report at least one group-held-out split that matches the intended use: by age if the goal is to predict later strength, and by mix or SCM family if the goal is to design new mixes. RMSE and mean error should be reported alongside R², because a single R² hides the systematic offset seen in Figure 2. For age specifically, the failure is structural, and more tuning will not give a tree a strength trend it has never seen. Physically motivated structure, such as a strength–maturity relationship, or a few measured late-age results for anchoring, is a more promising route."),
  H2("5.6  Limitations"),
  P("All results come from one laboratory dataset of 1,030 tests with seven of its eight original inputs, and nothing here was validated on field concrete. Hyperparameters were fixed rather than tuned. Tuning could narrow some composition gaps, but it cannot change the frozen age response shown in Figure 4. Three A3 bins are small (14 d: 62 rows; 56 d: 91; 180+ d: 59), so differences between models on those bins are not reliable. The three seeds capture model randomness, not uncertainty about which mixes happened to be tested. Finally, we did not run a leave-mix-out split, which Section 5.4 suggests would be the most direct test of generalization to new recipes."),
);

// 6 Conclusion
push(
  H1("6  Conclusion"),
  P("On the standard UCI concrete benchmark, tree models that reach R² 0.89–0.92 on random splits lose 0.25–0.80 in R², and their RMSE rises by a factor of 1.7 to 3, when the test set contains curing ages or mix ingredients missing from training. Predicting 28-day strength from early-age data fails completely. The age failure is a predictable offset: tree models stop at the last strength level they have seen. Random-split scores are also inflated because most test mixes already appear in training at another age. Group-held-out evaluation costs nothing to run on this dataset, and it gives a far more honest picture of how these models would perform in practice."),
);

// Back matter
push(
  H1("Data and code availability"),
  PL("All code, split definitions and result tables are available at https://github.com/RyanLatimer/concrete-logo. The data are loaded with *load_dataset(\"foundry-ml/dataset_concrete_compressive_strength\")* [11]. Running *experiments.py*, *eda.py*, *figures.py* and *analysis_deep.py* in that order regenerates every number and figure in this paper. Package versions are pinned in *requirements.txt* (Python 3.11, scikit-learn 1.9.1, XGBoost 3.4.2, SHAP 0.51.0)."),
  H1("Use of AI tools"),
  PL("Claude (Anthropic) assisted with analysis code, figure styling and drafting of the manuscript text. The author designed the study, ran the experiments, checked every reported number against the output files, and takes responsibility for the content."),
  H1("References"),
  ...REFS.map(refPara),
);

// ---------- document ----------
const doc = new Document({
  creator: "Ryan Latimer",
  title,
  styles: {
    default: { document: { run: { font: SERIF, size: BODY, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: SANS, size: 23, bold: true, color: INK },
        paragraph: { spacing: { before: 300, after: 110 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: SANS, size: 19, bold: true, color: INK2 },
        paragraph: { spacing: { before: 200, after: 70 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      titlePage: true,
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440, header: 620, footer: 620 } },
    },
    headers: {
      first: new Header({ children: [new Paragraph({ children: [] })] }),
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "Latimer · Random splits overstate concrete-strength prediction", font: SANS, size: 15, color: MUTED })] })] }),
    },
    footers: {
      first: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: SANS, size: 16, color: MUTED })] })] }),
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: SANS, size: 16, color: MUTED })] })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT); });
