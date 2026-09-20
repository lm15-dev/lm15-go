// Offline integration of the built WASM, Go Fetch transport and real loopback
// HTTP/SSE. Run: node cmd/lm15-wasm/smoke.mjs <wasm_exec.js> <lm15-go.wasm>
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { gzipSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';

const hostProcess = process;
const [execPath, wasmPath] = process.argv.slice(2);
await import(pathToFileURL(execPath).href);
const go = new Go();
const { instance } = await WebAssembly.instantiate(await readFile(wasmPath), go.importObject);
// Go disables Fetch specifically when it detects Node at package init. Hide
// Node during boot to exercise the browser transport, not its fake TCP stack.
globalThis.process = undefined;
const lifetime = go.run(instance);
while (!globalThis.lm15Go) await new Promise(r => setTimeout(r, 1));
globalThis.process = hostProcess;
lifetime.catch(err => { console.error(err); hostProcess.exit(1); });

let received = [];
let hangingClosed = false;
let streamFinished = false;
const server = createServer(async (req, res) => {
 const chunks = [];
 for await (const chunk of req) chunks.push(chunk);
 const body = Buffer.concat(chunks);
 received.push({ url: req.url, headers: req.headers, body });
 const data = JSON.parse(body);
 if (data.model === 'rate-limit') {
  res.writeHead(429, { 'content-type':'application/json', 'x-request-id':'req-offline', 'retry-after':'2', 'x-ratelimit-remaining-requests':'0', 'authorization':'do-not-retain' });
  res.end(JSON.stringify({error:{message:'quota',code:'rate_limit'}}));
 } else if (data.model === 'hang') {
  res.writeHead(200, {'content-type':'text/event-stream'});
  res.write('data: '+JSON.stringify({model:'hang',choices:[{delta:{content:'first'},finish_reason:null}]})+'\n\n');
  res.on('close', () => { hangingClosed = true; });
 } else if (req.url === '/v1/messages') {
  assert.equal(req.headers['anthropic-dangerous-direct-browser-access'], 'true');
  res.setHeader('content-type','application/json');
  res.end(JSON.stringify({id:'a',model:'m',role:'assistant',content:[{type:'text',text:'anthropic'}],stop_reason:'end_turn',usage:{input_tokens:1,output_tokens:1}}));
 } else if (data.stream) {
  res.writeHead(200, {'content-type':'text/event-stream'});
  res.write('data: '+JSON.stringify({id:'s',model:'m',choices:[{delta:{content:'first'},finish_reason:null}]})+'\n\n');
  setTimeout(() => {
   streamFinished = true;
   res.end('data: '+JSON.stringify({choices:[{delta:{content:' second'},finish_reason:'stop'}]})+'\n\ndata: [DONE]\n\n');
  }, 80);
 } else {
  const response = {id:'r',model:'m',choices:[{message:{role:'assistant',content:'hello'},finish_reason:'stop'}]};
  res.writeHead(200, {'content-type':'application/json','content-encoding':'gzip'});
  res.end(gzipSync(JSON.stringify(response)));
 }
});
await new Promise(r => server.listen(0,'127.0.0.1',r));
const base_url = `http://127.0.0.1:${server.address().port}`;
const input = (model='m',provider='openai-chat') => ({provider,api_key:'offline-placeholder',base_url,canonical_request:{model,messages:[{role:'user',parts:[{type:'text',text:'hello'}]}]}});
const call = async (op, value, ...rest) => JSON.parse(await lm15Go.call(op, JSON.stringify(value), ...rest));
const deadline = setTimeout(() => { console.error('WASM smoke timed out'); hostProcess.exit(1); }, 10000);
try {
 assert.equal((await call('version',{})).language,'go');
 const built = await call('build_request',input());
 assert.ok(built.body_b64);
 assert.equal(received.length,0,'build_request must not touch network');
 const complete = await call('complete',input());
 assert.ok(complete.canonical_response, JSON.stringify(complete));
 assert.deepEqual(received[0].body,Buffer.from(built.body_b64,'base64'));
 assert.equal(complete.canonical_response.message.parts[0].text,'hello');
 const events = [];
 const streamed = await call('stream',input(),undefined,raw => {
  const event = JSON.parse(raw); events.push(event);
  if (event.type === 'delta' && events.filter(e=>e.type==='delta').length === 1) assert.equal(streamFinished,false,'event was buffered');
 });
 assert.equal(streamed.canonical_response.message.parts[0].text,'first second');
 assert.ok(events.length >= 3);
 const anthropic = await call('complete',input('m','anthropic'));
 assert.ok(anthropic.canonical_response, JSON.stringify(anthropic));
 const failed = await call('complete',input('rate-limit'));
 assert.equal(failed.error.name,'RateLimitError');
 assert.equal(failed.error.http_response.request_id,'req-offline');
 assert.equal(failed.error.http_response.retry_after,2);
 assert.ok(failed.error.message.includes('req-offline'));
 assert.ok(!JSON.stringify(failed).includes('do-not-retain'));
 for (const args of [[],[1,'{}'],['complete','{'],['complete','null'],['complete','{}',false],['stream',JSON.stringify(input()),undefined,1]]) {
  assert.ok(JSON.parse(await lm15Go.call(...args)).error);
 }
 // Listener accounting and abort AFTER headers/body data: this catches Go's
 // Fetch stream reader not automatically watching the context after RoundTrip.
 const ac = new AbortController();
 let attached = 0;
 const signal = {get aborted(){return ac.signal.aborted;},addEventListener(...args){attached++;ac.signal.addEventListener(...args);},removeEventListener(...args){attached--;ac.signal.removeEventListener(...args);}};
 const aborted = await call('stream',input('hang'),signal,raw => {
  if (JSON.parse(raw).type==='delta') setTimeout(()=>ac.abort(),10);
 });
 assert.equal(aborted.error.name,'AbortError');
 assert.equal(attached,0);
 await new Promise(r=>setTimeout(r,30));
 assert.equal(hangingClosed,true,'aborting must close the HTTP stream');
 const before = received.length;
 assert.equal((await call('complete',input(),ac.signal)).error.name,'AbortError');
 assert.equal(received.length,before);
 const callbackFailure = await call('stream',input(),undefined,()=>{throw new Error('host callback error');});
 assert.ok(callbackFailure.error);
 // A bad host callback must not kill the shared Go runtime.
 assert.equal((await call('version',{})).language,'go');
 console.log('PASS: WASM boot, real SDK bytes/Fetch, gzip, incremental stream/assembly, Anthropic opt-in, diagnostics, invalid inputs, abort + listener cleanup, callback recovery');
} finally {
 clearTimeout(deadline);
 server.closeAllConnections();
 server.close();
}
hostProcess.exit(0);
