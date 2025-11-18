import React, { useState } from 'react';
import axios from 'axios';
import { Container, InputGroup, FormControl, Button, Card, Spinner } from 'react-bootstrap';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function App() {
  const [query, setQuery] = useState('');
  const [summary, setSummary] = useState('');
  const [chartData, setChartData] = useState(null);
  const [tableData, setTableData] = useState([]);
  const [loading, setLoading] = useState(false);
  const backendBase = 'http://127.0.0.1:8000'; // change if your backend uses different host/port

  const sendQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setSummary('');
    setChartData(null);
    setTableData([]);

    try {
      const res = await axios.post(`${backendBase}/api/query/`, { query }, { timeout: 20000 });
      const data = res.data || {};

      // summary (string)
      setSummary(data.summary || 'No summary returned.');

      // chart: expect [{year: 2020, price: 1000000, demand: 5}, ...]
      if (Array.isArray(data.chart) && data.chart.length > 0) {
        // Use 'year' as labels and include numeric series that exist
        const labels = data.chart.map(r => r.year);
        const datasets = [];
        if (Object.prototype.hasOwnProperty.call(data.chart[0], 'price')) {
          datasets.push({
            label: 'Price',
            data: data.chart.map(r => r.price ?? null),
            tension: 0.3,
            fill: false,
          });
        }
        if (Object.prototype.hasOwnProperty.call(data.chart[0], 'demand')) {
          datasets.push({
            label: 'Demand',
            data: data.chart.map(r => r.demand ?? null),
            tension: 0.3,
            fill: false,
          });
        }
        setChartData({ labels, datasets });
      } else {
        setChartData(null);
      }

      // table: expect list of objects
      if (Array.isArray(data.table) && data.table.length > 0) {
        setTableData(data.table);
      } else {
        setTableData([]);
      }
    } catch (err) {
      console.error(err);
      setSummary('Error contacting server. Is backend running at ' + backendBase + '?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="p-3" style={{ maxWidth: 900 }}>
      <h3 className="mb-3">Real Estate Chat (frontend)</h3>

      <InputGroup className="mb-3">
        <FormControl
          placeholder='Type a query, e.g. "Analyze Wakad"'
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') sendQuery(); }}
        />
        <Button variant="primary" onClick={sendQuery} disabled={loading}>
          {loading ? <Spinner as="span" animation="border" size="sm" /> : 'Send'}
        </Button>
      </InputGroup>

      <Card className="mb-3">
        <Card.Body>
          <strong>Summary</strong>
          <div style={{ whiteSpace: 'pre-wrap', marginTop: 8 }}>{summary || 'No query yet.'}</div>
        </Card.Body>
      </Card>

      {chartData && (
        <Card className="mb-3">
          <Card.Body>
            <h5>Trends</h5>
            <div style={{ height: 300 }}>
              <Line data={chartData} />
            </div>
          </Card.Body>
        </Card>
      )}

      {tableData.length > 0 && (
        <Card className="mb-3">
          <Card.Body>
            <h5>Data (first 50 rows)</h5>
            <div style={{ overflowX: 'auto' }}>
              <table className="table table-sm">
                <thead>
                  <tr>
                    {Object.keys(tableData[0]).map((k) => <th key={k}>{k}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {tableData.slice(0, 50).map((row, i) => (
                    <tr key={i}>
                      {Object.keys(row).map((k) => <td key={k}>{String(row[k])}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card.Body>
        </Card>
      )}
    </Container>
  );
}

export default App;
