fetch("data/leaderboard.tsv")
  .then(res => res.text())
  .then(txt => {
    const rows = txt.trim().split("\n").map(r => r.split("\t"));
    const header = rows[0];
    const body = rows.slice(1);

    let html = "<thead><tr>";
    header.forEach(h => html += `<th>${h}</th>`);
    html += "</tr></thead><tbody>";

    body.forEach(r => {
      html += "<tr>";
      r.forEach(c => html += `<td>${c}</td>`);
      html += "</tr>";
    });
    html += "</tbody>";

    document.getElementById("table").innerHTML = html;
  });
