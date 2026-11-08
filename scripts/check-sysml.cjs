// SPDX-License-Identifier: Apache-2.0
// Parse and check a SysML v2 textual model with the FLOSS language server `sysml-v2-lsp` (MIT).
//   npm install sysml-v2-lsp@0.42.0 --ignore-scripts      (in any scratch directory)
//   node check-sysml.cjs ../model/AndroidHPC.sysml  [--lsp-dir <dir containing node_modules>]
// Exit code 0 = no errors, 1 = errors, 2 = no diagnostics were published (could not tell).
// The server publishes several times (syntax first, semantic checks last), so this waits until
// publishing has been quiet for 15 s and reports only the last publish. A syntax-only answer is
// not a pass. sysml-v2-lsp is an independent ANTLR implementation, not the OMG reference.
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const li = args.indexOf('--lsp-dir');
const lspDir = li >= 0 ? path.resolve(args.splice(li, 2)[1]) : process.cwd();
if (!args[0]) { console.error('usage: node check-sysml.cjs MODEL.sysml [--lsp-dir DIR]'); process.exit(2); }
const file = path.resolve(args[0]);
const server = path.join(lspDir, 'node_modules', 'sysml-v2-lsp', 'dist', 'server', 'server.js');
if (!fs.existsSync(server)) { console.error('sysml-v2-lsp not found under ' + lspDir); process.exit(2); }

const text = fs.readFileSync(file, 'utf8');
const uri = 'file:///' + file.replace(/\\/g, '/').replace(/^\//, '');
const p = spawn(process.execPath, [server, '--stdio'], { stdio: ['pipe', 'pipe', 'ignore'] });
let buf = Buffer.alloc(0);
let diags = [], published = 0, quiet, done = false;
const send = (o) => { const s = JSON.stringify(o); p.stdin.write(`Content-Length: ${Buffer.byteLength(s)}\r\n\r\n${s}`); };

function finish() {
  if (done) return; done = true;
  const names = ['', 'ERROR', 'WARNING', 'INFO', 'HINT'];
  const count = [0, 0, 0, 0, 0];
  for (const d of diags) {
    const sev = d.severity || 1; count[sev]++;
    if (sev <= 2) console.log(`${names[sev]} ${path.basename(file)}:${d.range.start.line + 1}:${d.range.start.character + 1} ${d.message}`);
  }
  console.log(`errors=${count[1]} warnings=${count[2]} info=${count[3]} hints=${count[4]} publishes=${published}`);
  p.kill();
  process.exit(published === 0 ? 2 : count[1] > 0 ? 1 : 0);
}

p.stdout.on('data', (d) => {
  buf = Buffer.concat([buf, d]);
  for (;;) {
    const i = buf.indexOf('\r\n\r\n'); if (i < 0) return;
    const m = /Content-Length: (\d+)/i.exec(buf.slice(0, i).toString()); if (!m) return;
    const len = +m[1]; if (buf.length < i + 4 + len) return;
    const msg = JSON.parse(buf.slice(i + 4, i + 4 + len).toString());
    buf = buf.slice(i + 4 + len);
    if (msg.id === 1) {
      send({ jsonrpc: '2.0', method: 'initialized', params: {} });
      send({ jsonrpc: '2.0', method: 'textDocument/didOpen', params: { textDocument: { uri, languageId: 'sysml', version: 1, text } } });
    } else if (msg.method === 'workspace/configuration') {
      send({ jsonrpc: '2.0', id: msg.id, result: (msg.params.items || []).map(() => ({})) });
    } else if (msg.method === 'client/registerCapability' || msg.method === 'window/workDoneProgress/create') {
      send({ jsonrpc: '2.0', id: msg.id, result: null });
    } else if (msg.method === 'textDocument/publishDiagnostics' && msg.params.uri.toLowerCase().endsWith(path.basename(file).toLowerCase())) {
      published++; diags = msg.params.diagnostics;
      clearTimeout(quiet); quiet = setTimeout(finish, 15000);
    }
  }
});
send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { processId: process.pid, rootUri: null, capabilities: {}, workspaceFolders: null } });
setTimeout(finish, 90000);
