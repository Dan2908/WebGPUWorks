const GRID_SIZE = 64;
const NUM_CELLS = GRID_SIZE * GRID_SIZE;
const UPDATE_INTERVAL = 200;
const WORKGROUP_SIZE = 8;

let LOGGER_COUNT = 0;

const VERTEX_DATA = {
	rect_vertices: new Float32Array([	/* Vertex array */
		-0.8, 0.8,
		0.8, 0.8,
		0.8, -0.8,

		-0.8, 0.8,
		0.8, -0.8,
		-0.8, -0.8
	]),
	gridSizeUniform: new Float32Array([GRID_SIZE, GRID_SIZE]),
	cellStateArray: new Uint32Array(GRID_SIZE * (GRID_SIZE / 32))
};
///----------------------------

// Log message in the log box. "type" can be set to "error", "warning" or "info" to color it accordingly.
function logMsg(msg, type) {

	const loggerLine = document.createElement('div');

	const displayMsg = document.createElement('p');
	displayMsg.setAttribute("class", type);
	displayMsg.innerText = "[" + type + "] " + msg;

	const loggerCount = document.createElement('p');
	loggerCount.setAttribute("class", "log_count");
	loggerCount.innerText = LOGGER_COUNT + ":";

	loggerLine.appendChild(loggerCount);
	loggerLine.appendChild(displayMsg);

	document.getElementById('messages').appendChild(loggerLine);

	++LOGGER_COUNT;
}

// Get the 32 bit binary String representation of the given integer, in groups of four.
function BinRep(number) {
	let binNum = number.toString(2);
	const leftZeros = 32 - binNum.length;

	binNum = String("").padStart(leftZeros, "0") + binNum;
	let output = "";
	for (let q = 0; q < 32; q += 4) {
		output += binNum.substring(q, q + 4) + " ";
	}

	return output;
}

// Get the GPU device.
async function getAdapterDevice() {
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		logMsg("Error", "Couldn't get navigator's GPU Adapter.", "error");
	}

	const device = await adapter.requestDevice();
	if (!device) {
		logMsg("Couldn't get GPU Device.", "error");
	}

	return device;
}

function generateCells()
{
	for (let i = 0; i < VERTEX_DATA.cellStateArray.length; ++i) {
		const random = Math.trunc(Math.random() * 100)
		VERTEX_DATA.cellStateArray[i] += (8 << (random%32));
	};
}

// Start working
async function start() {

	// check navigator support
	if (!navigator.gpu) {
		logMsg("WebGPU is not supported by this browser.", "error");
	}

	// Get Device
	const device = await getAdapterDevice();

	// Setup canvas
	const canvasElement = document.getElementById("canvas_box");

	const context = canvasElement.getContext("webgpu");
	const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
	context.configure({
		device: device,
		format: canvasFormat
	});

	// -------- CreateBuffers
	/* Vertex buffer */
	const vertexBuffer = device.createBuffer({
		label: "Vertex buffer",
		size: VERTEX_DATA.rect_vertices.byteLength,
		usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(vertexBuffer, 0, VERTEX_DATA.rect_vertices);	/* Vertex array -> Buffer */

	/* Uniform buffer */
	const uniformBuffer = device.createBuffer({
		label: "Uniform Buffer",
		size: VERTEX_DATA.gridSizeUniform.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(uniformBuffer, 0, VERTEX_DATA.gridSizeUniform);	/* Uniform array -> Buffer */

	/* cell state (storage) buffer */
	const cellStateBuffer = [
		device.createBuffer({
			label: "Cell State A",
			size: VERTEX_DATA.cellStateArray.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		}),
		device.createBuffer({	/* cell state (storage) buffer */
			label: "Cell State B",
			size: VERTEX_DATA.cellStateArray.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		})];

	//Randomly generate a 32bit number to randomly activate cells
	generateCells();
	device.queue.writeBuffer(cellStateBuffer[0], 0, VERTEX_DATA.cellStateArray); /* Cell state array -> Buffer A*/

	//Once again to the second buffer
	//generateCells();
	device.queue.writeBuffer(cellStateBuffer[1], 0, VERTEX_DATA.cellStateArray); /* Cell state array -> Buffer B*/


	const cellShaderModule = device.createShaderModule({
		label: "Cell Shader Module",
		code: `
		struct VertexInput {
			@location(0) pos: vec2f, 
			@builtin(instance_index) instance: u32
		};
		
		struct VertexOutput {
			@builtin(position) pos: vec4f, 
			@location(0) cell: vec2f
		};
		
		struct FragInput {
			@location(0) cell: vec2f
		};
		
		@group(0) @binding(0) var<uniform> grid: vec2f;
		@group(0) @binding(1) var<storage> cellState: array<u32>;
		
		fn getState(index: u32) -> f32 {
			let byte = u32(index/32);
			let bit = u32(index%32);
			return f32((cellState[byte] >> bit) & 1u);
		}
		
		@vertex 
		fn vertexMain(input: VertexInput) -> VertexOutput {	
			let cellIndex = f32(input.instance);
			let cell = vec2f(cellIndex % grid.x, floor(cellIndex / grid.x));
			let cellOffset = cell / grid * 2;
			let gridPos = (input.pos + 1) / grid - 1 + cellOffset;
			
			var output: VertexOutput;
			output.pos = getState(input.instance) * vec4f(gridPos, 0, 1);
			output.cell = cell;
			return output;
		}
		
		@fragment
		fn fragmentMain(input: FragInput) -> @location(0) vec4f {
			let c = input.cell / grid;
			return vec4f(c, 1-c.x, 1);
		}
		`
	});

	const simulationShaderModule = device.createShaderModule({
		label: "Simulation Shader Module.",
		code: `
		@group(0) @binding(0) var<uniform> grid: vec2f;
		
		@group(0) @binding(1) var<storage> cellStateIn: array<u32>;
		@group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;
		
		fn getRaw(cell: vec3u) -> u32 {
			return (cell.x % u32(grid.x)) + (cell.y % u32(grid.y)) * u32(grid.x);
		}

		fn getState(cell: vec3u) -> u32 {
			let raw = getRaw(cell);
			let byte = raw/32;
			let bit = raw%32;
			return (cellStateIn[byte] >> bit) & 1u;
		}

		fn setState(cell: vec3u, state: bool) {
			let raw = getRaw(cell);
			let byte = raw/32;
			let bit = raw%32;
			if(state){
				cellStateOut[byte] |= (1u << bit);
			} else {
				cellStateOut[byte] &= ~(1u << bit);
			}
		}

		@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
		fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
			let activeNgbs = getState(vec3u(cell.x    , cell.y - 1, cell.z)) +
							 getState(vec3u(cell.x    , cell.y + 1, cell.z)) +
							 getState(vec3u(cell.x - 1, cell.y - 1, cell.z)) +
							 getState(vec3u(cell.x - 1, cell.y    , cell.z)) +
							 getState(vec3u(cell.x - 1, cell.y + 1, cell.z)) +
							 getState(vec3u(cell.x + 1, cell.y - 1, cell.z)) +
							 getState(vec3u(cell.x + 1, cell.y    , cell.z)) +
							 getState(vec3u(cell.x + 1, cell.y + 1, cell.z));

			// Conway's game of life rules:
			switch activeNgbs {
				case 2: {
					setState(cell, getState(cell) == 1);
				}
				case 3: {
					setState(cell, true);
				}
				default: {
					setState(cell, false);
				}
			}
		}
		`
	});

	const vertexBufferLayout = {
		arrayStride: 8,
		attributes: [{
			format: "float32x2",
			offset: 0,
			shaderLocation: 0
		}]
	}

	const bindGroupLayout = device.createBindGroupLayout({
		label: "Cell Bind Group Layout",
		entries: [{
			binding: 0,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
			buffer: {} // defaults to "type: uniform"
		}, {
			binding: 1,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
			buffer: { type: "read-only-storage" }
		}, {
			binding: 2,
			visibility: GPUShaderStage.COMPUTE,
			buffer: { type: "storage" }
		}]
	});

	const bindGroups = [
		device.createBindGroup({
			label: "Bind Group A",
			layout: bindGroupLayout,
			entries: [{
				binding: 0,
				resource: { buffer: uniformBuffer }
			}, {
				binding: 1,
				resource: { buffer: cellStateBuffer[0] }
			}, {
				binding: 2,
				resource: { buffer: cellStateBuffer[1] }
			}]
		}),
		device.createBindGroup({
			label: "Bind Group B",
			layout: bindGroupLayout,
			entries: [{
				binding: 0,
				resource: { buffer: uniformBuffer }
			}, {
				binding: 1,
				resource: { buffer: cellStateBuffer[1] }
			}, {
				binding: 2,
				resource: { buffer: cellStateBuffer[0] }
			}]
		})
	];

	const pipelineLayout = device.createPipelineLayout({
		label: "Cell Pipeline Layout",
		bindGroupLayouts: [bindGroupLayout]
	});

	const cellPipeline = device.createRenderPipeline({
		label: "Cell Pipeline",
		layout: pipelineLayout,
		vertex: {
			module: cellShaderModule,
			entryPoint: "vertexMain",
			buffers: [vertexBufferLayout]
		},
		fragment: {
			module: cellShaderModule,
			entryPoint: "fragmentMain",
			targets: [{
				format: canvasFormat
			}]
		}
	});

	const simulationPipeline = device.createComputePipeline({
		label: "Simulation pipeline",
		layout: pipelineLayout,
		compute: {
			module: simulationShaderModule,
			entryPoint: "computeMain"
		}
	});

	// Loop
	let step = 0;
	function UpdateTable() {
		const encoder = device.createCommandEncoder();
		// Compute Pass
		const computePass = encoder.beginComputePass();

		computePass.setPipeline(simulationPipeline);
		computePass.setBindGroup(0, bindGroups[step % 2]);
		const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
		computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

		computePass.end();

		++step;
		// Render Pass
		const pass = encoder.beginRenderPass({
			colorAttachments: [{
				view: context.getCurrentTexture().createView(),
				loadOp: "clear",
				clearValue: [0, 0, 0, 1],
				storeOp: "store"
			}]
		})

		// Draw
		pass.setPipeline(cellPipeline);
		pass.setBindGroup(0, bindGroups[step % 2]);
		pass.setVertexBuffer(0, vertexBuffer);
		pass.draw(VERTEX_DATA.rect_vertices.length / 2, NUM_CELLS);

		pass.end();
		device.queue.submit([encoder.finish()]); /* encoder.finish() returns the command buffer to submit */
	}

	setInterval(UpdateTable, UPDATE_INTERVAL);
}

try {
	logMsg("A long enough message to test the x overflow behavior in x, meaning the horizontal overflow of this box. We will try to fit more text than a normal screen can hold in a single line, then check if it is shown as expected.", "warning")
	start()
}
catch (e) {
	logMsg(("start() - Uncaught JavaScript error.", $(e)), "error");
}