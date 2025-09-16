import React from "react";
import { Page, Text, View, Document, StyleSheet } from "@react-pdf/renderer";

// Improved PDF styles
const styles = StyleSheet.create({
  page: {
    padding: 24,
    fontFamily: "Helvetica",
    backgroundColor: "#e0f2fe",
    color: "#1e3a8a",
  },
  header: {
    fontSize: 28,
    fontWeight: "bold",
    marginBottom: 8,
    color: "#2563eb",
    textAlign: "center",
    textDecoration: "underline",
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 16,
    color: "#2563eb",
    textAlign: "center",
  },
  section: {
    marginBottom: 18,
    padding: 12,
    borderRadius: 8,
    backgroundColor: "#fff",
    boxShadow: "0 2px 8px #60a5fa22",
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 8,
    color: "#2563eb",
    textDecoration: "underline",
  },
  table: {
    display: "table",
    width: "auto",
    marginVertical: 8,
    borderStyle: "solid",
    borderWidth: 1,
    borderColor: "#60a5fa",
    borderRadius: 4,
    overflow: "hidden",
  },
  tableRow: {
    flexDirection: "row",
  },
  tableHeader: {
    backgroundColor: "#e0f2fe",
    fontWeight: "bold",
  },
  tableCell: {
    padding: 6,
    borderRightWidth: 1,
    borderBottomWidth: 1,
    borderColor: "#60a5fa",
    minWidth: 90,
    fontSize: 12,
  },
  tableCellLast: {
    padding: 6,
    borderBottomWidth: 1,
    borderColor: "#60a5fa",
    minWidth: 120,
    fontSize: 12,
  },
  badgeNovel: {
    backgroundColor: "#fde68a",
    color: "#92400e",
    padding: "2 4",
    borderRadius: 4,
    fontSize: 10,
  },
  badgeKnown: {
    backgroundColor: "#bbf7d0",
    color: "#065f46",
    padding: "2 4",
    borderRadius: 4,
    fontSize: 10,
  },
  small: {
    fontSize: 10,
    color: "#64748b",
  },
  taxonomyList: {
    marginTop: 6,
    marginLeft: 12,
    fontSize: 12,
    color: "#1e3a8a",
  },
  vizList: {
    marginTop: 6,
    marginLeft: 12,
    fontSize: 12,
    color: "#1e3a8a",
  },
});

const getTaxonomySummary = (results) => {
  // Get unique values for each taxonomy field
  const fields = ["kingdom", "phylum", "class", "order", "family", "genus"];
  const summary = {};
  fields.forEach((field) => {
    summary[field] = Array.from(
      new Set(results.map((r) => r[field]).filter((v) => v && v !== "Unknown"))
    );
  });
  return summary;
};

const DashboardPDF = ({ data }) => {
  const taxonomy = getTaxonomySummary(data.results);

  return (
    <Document>
      <Page style={styles.page}>
        {/* Header */}
        <Text style={styles.header}>Biodiversity Dashboard Report</Text>
        <Text style={styles.subtitle}>Analyze and visualize ecological data with precision</Text>

        {/* Overview Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Overview</Text>
          <Text>Total Sequences: {data.results.length}</Text>
          <Text>
            Unique Species: {new Set(data.results.map(r => r.predicted_species)).size}
          </Text>
          <Text>
            Alpha Diversity (Richness): {data.alpha_diversity?.richness ?? "N/A"}
          </Text>
          <Text>
            Shannon Index: {data.alpha_diversity?.shannon_index ?? "N/A"}
          </Text>
        </View>

        {/* Taxonomic Analysis Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Taxonomic Analysis</Text>
          {Object.keys(taxonomy).map((field) => (
            <Text key={field} style={styles.taxonomyList}>
              <Text style={{ fontWeight: "bold" }}>{field.charAt(0).toUpperCase() + field.slice(1)}:</Text>{" "}
              {taxonomy[field].length > 0 ? taxonomy[field].join(", ") : "Unknown"}
            </Text>
          ))}
        </View>

        {/* Data Visualization Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Data Visualization</Text>
          <Text style={styles.vizList}>
            <Text style={{ fontWeight: "bold" }}>Alpha Diversity:</Text>{" "}
            Richness: {data.alpha_diversity?.richness ?? "N/A"}, Shannon Index: {data.alpha_diversity?.shannon_index ?? "N/A"}, Evenness: {data.alpha_diversity?.evenness ?? "N/A"}
          </Text>
          {data.rarefaction_curve && Object.keys(data.rarefaction_curve).length > 0 && (
            <Text style={styles.vizList}>
              <Text style={{ fontWeight: "bold" }}>Rarefaction Curve:</Text>{" "}
              {JSON.stringify(data.rarefaction_curve)}
            </Text>
          )}
          {data.beta_diversity && Object.keys(data.beta_diversity).length > 0 && (
            <Text style={styles.vizList}>
              <Text style={{ fontWeight: "bold" }}>Beta Diversity:</Text>{" "}
              {JSON.stringify(data.beta_diversity)}
            </Text>
          )}
        </View>

        {/* Prediction Results Table */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Prediction Results</Text>
          <View style={styles.table}>
            <View style={[styles.tableRow, styles.tableHeader]}>
              <Text style={styles.tableCell}>Predicted Species</Text>
              <Text style={styles.tableCell}>Confidence</Text>
              <Text style={styles.tableCell}>Status</Text>
              <Text style={styles.tableCell}>Cluster ID</Text>
              <Text style={styles.tableCellLast}>Top 3 Predictions</Text>
            </View>
            {data.results.map((item, idx) => (
              <View style={styles.tableRow} key={idx}>
                <Text style={styles.tableCell}>{item.predicted_species || "-"}</Text>
                <Text style={styles.tableCell}>
                  {item.confidence ? (item.confidence * 100).toFixed(2) + "%" : "-"}
                </Text>
                <Text style={styles.tableCell}>
                  <Text style={item.status === "novel" ? styles.badgeNovel : styles.badgeKnown}>
                    {item.status || "-"}
                  </Text>
                </Text>
                <Text style={styles.tableCell}>{item.cluster_id || "-"}</Text>
                <Text style={styles.tableCellLast}>
                  {(item.top_predictions || [])
                    .slice(0, 3)
                    .map(tp => `${tp.species} (${(tp.confidence * 100).toFixed(2)}%)`)
                    .join("; ")}
                </Text>
              </View>
            ))}
          </View>
        </View>

        {/* Footer */}
        <Text style={[styles.small, { textAlign: "center", marginTop: 24 }]}>
          Generated on {new Date().toLocaleDateString()}
        </Text>
      </Page>
    </Document>
  );
};

export default DashboardPDF;