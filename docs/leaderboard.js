fetch("data/leaderboard.tsv")
  .then(res => res.text())
  .then(txt => {
    const rows = txt.trim().split("\n").map(r => r.split("\t"));
    const header = rows[0];
    const body = rows.slice(1);

    // Sort by a main metric if present
    const mainMetric = header.find(h => h.includes("val_auroc")) || header[0];
    const mIdx = header.indexOf(mainMetric);

    body.sort((a, b) => {
      const x = parseFloat(a[mIdx]);
      const y = parseFloat(b[mIdx]);
      if (isNaN(x) || isNaN(y)) return 0;
      return y - x;
    });

    let html = "<thead><tr>";
    header.forEach(h => html += `<th>${h}</th>`);
    html += "</tr></thead><tbody>";

    body.forEach(r => {
      html += "<tr>";
      r.forEach(c => {
        let v = c;
        if (!isNaN(c) && c !== "") {
          v = Number(c).toFixed(4);
        }
        html += `<td>${v}</td>`;
      });
      html += "</tr>";
    });
    html += "</tbody>";

    document.getElementById("table").innerHTML = html;
  })
  .catch(err => {
    console.error("Leaderboard load failed:", err);
  });
