fetch("data/leaderboard.tsv")
  .then(r => r.text())
  .then(txt => {
    const rows = txt.trim().split("\n").map(r => r.split("\t"));
    const header = rows[0];
    const data = rows.slice(1).map(r => {
      let o = {};
      header.forEach((h, i) => o[h] = r[i]);
      return o;
    });

    const filters = {
      data: document.getElementById("filter-dataset"),
      model: document.getElementById("filter-model"),
      explainer: document.getElementById("filter-explainer")
    };

    function populateInputs(container, key, type = "checkbox") {
      const vals = [...new Set(data.map(d => d[key]))].sort();
      const name = `filter-${key}`;

      container.innerHTML = vals.map((v, i) =>
        `<label>
          <input type="${type}" name="${name}" value="${v}" ${i === 0 || type === "checkbox" ? "checked" : ""}>
          ${v}
        </label>`
      ).join("");

      container.querySelectorAll("input").forEach(inp => inp.onchange = render);
    }


    populateInputs(filters.data, "data", "radio");     
    populateInputs(filters.model, "model", "checkbox");
    populateInputs(filters.explainer, "explainer", "checkbox");


    function selected(container) {
      return [...container.querySelectorAll("input:checked")].map(cb => cb.value);
    }

    function filteredData() {
      const ds = selected(filters.data);
      const ms = selected(filters.model);
      const es = selected(filters.explainer);
      return data.filter(d =>
        ds.includes(d.data) &&
        ms.includes(d.model) &&
        es.includes(d.explainer)
      );
    }

    /* -------- TABLE -------- */

    function renderTable(rows) {
      const cols = [
        "data","model","explainer",
        "val_precision", 
        "val_recall",
        "val_f1",
        "val_auroc",
        "metric_faithfulness_drop@0.20",
        "metric_comp@0.20",
        "metric_suff@0.20"
      ].filter(c => header.includes(c));

      let html = "<thead><tr>";
      cols.forEach(c => html += `<th>${c}</th>`);
      html += "</tr></thead><tbody>";

      rows.forEach(r => {
        html += "<tr>";
        cols.forEach(c => {
          let v = r[c];
          if (!isNaN(v)) v = Number(v).toFixed(4);
          html += `<td>${v ?? ""}</td>`;
        });
        html += "</tr>";
      });

      html += "</tbody>";
      document.getElementById("table").innerHTML = html;
    }

    /* -------- PLOTS -------- */

    function plotCurves(divId, title, prefix, ylabel) {
      const rows = filteredData();
      const expls = [...new Set(rows.map(r => r.explainer))];

      const cols = header
        .filter(h => h.startsWith(prefix + "@"))
        .map(h => ({ h, k: parseFloat(h.split("@")[1]) }))
        .sort((a,b)=>a.k-b.k);

      const traces = expls.map(expl => {
        const rs = rows.filter(r => r.explainer === expl);
        return {
          name: expl,
          x: cols.map(c => c.k),
          y: cols.map(c =>
            rs.map(r => parseFloat(r[c.h]))
              .filter(v => !isNaN(v))
              .reduce((a,b)=>a+b,0) / rs.length
          ),
          mode: "lines+markers",
          line: { width: 2 }
        };
      });

      Plotly.newPlot(divId, traces, {
        title,
        font: { family: "Georgia, serif" },
        xaxis: { title: "Fraction k", gridcolor: "#eee" },
        yaxis: { title: ylabel, gridcolor: "#eee" },
        legend: { orientation: "h" },
        margin: { t: 40, l: 60, r: 20, b: 40 }
      }, { displayModeBar: false });
    }

    function render() {
      const rows = filteredData();
      renderTable(rows);

      plotCurves(
        "plot-faithfulness",
        "Faithfulness vs k",
        "metric_faithfulness_drop",
        "Probability drop"
      );

      plotCurves(
        "plot-comprehensiveness",
        "Comprehensiveness vs k",
        "metric_comp",
        "Probability increase"
      );

      plotCurves(
        "plot-sufficiency",
        "Sufficiency vs k",
        "metric_suff",
        "Probability drop"
      );
    }

    render();
  });
