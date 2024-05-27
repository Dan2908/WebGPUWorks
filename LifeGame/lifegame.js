const GRID_SIZE = 32;
const NUM_CELLS = GRID_SIZE * GRID_SIZE;
const UPDATE_INTERVAL = 200;
const WORKGROUP_SIZE = 8;

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
	cellStateArray: new Uint32Array(NUM_CELLS / 32)
};
///----------------------------

function printError(msg) {
	const errorBox = document.getElementById('error_box');
	const errorMsg = document.createElement('p')
	errorBox.hidden = false;

	errorMsg.innerText = msg;
	errorBox.appendChild(errorMsg);
}

function hideErrorBox() {
	document.getElementById('error_box').hidden = true;
}

async function getAdapterDevice() {
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		printError("Couldn't get navigator's GPU Adapter.");
	}

	const device = await adapter.requestDevice();
	if (!device) {
		printError("Couldn't get GPU Device.");
	}

	return device;
}

async function start() {

	// Clear error box
	hideErrorBox();
	// check navigator support
	if (!navigator.gpu) {
		printError("WebGPU is not supported by this browser.");
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
	VERTEX_DATA.cellStateArray.forEach(element => {
		element = Math.trunc(Math.random() * 0xFFFFFFFF);
	});
	device.queue.writeBuffer(cellStateBuffer[0], 0, VERTEX_DATA.cellStateArray); /* Cell state array -> Buffer A*/

	//Once again to the second buffer
	VERTEX_DATA.cellStateArray.forEach(element => {
		element = Math.trunc(Math.random() * 0xFFFFFFFF);
	});
	device.queue.writeBuffer(cellStateBuffer[1], 0, VERTEX_DATA.cellStateArray); /* Cell state array -> Buffer B*/

	const vertexBufferLayout = {
		arrayStride: 8,
		attributes: [{
			format: "float32x2",
			offset: 0,
			shaderLocation: 0
		}]
	}

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
		
		fn getCellState(index: u32) -> f32 {
			if((cellState[index/32] & u32(1 << (index%32))) != 0){
				return 1;
			} 
			return 0;
		}
		
		@vertex 
		fn vertexMain(input: VertexInput) -> VertexOutput {
			
			let cellIndex = f32(input.instance);
			let cell = vec2f(cellIndex % grid.x, floor(cellIndex / grid.x));
			let cellOffset = cell / grid * 2;
			let gridPos = (getCellState(input.instance) * input.pos + 1) / grid - 1 + cellOffset;
			
			var output: VertexOutput;
			output.pos = vec4f(gridPos, 0, 1);
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
		code:`
		@group(0) @binding(0) var<uniform> grid: vec2f;
		
		@group(0) @binding(1) var<storage> cellStateIn: array<u32>;
		@group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

		fn getCellIndex(cell: vec2u) -> u32{
			return (cell.y * u32(grid.x) + cell.x);
		}

		fn switchCellState(index: u32) {
			cellStateOut[index/32] ^= u32(1 << (index%32));
		}

		@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
		fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
			let index = getCellIndex(cell.xy);
			switchCellState(index);
		}
		`
	});

	const bindGroupLayout = device.createBindGroupLayout({
		label: "Cell Bind Group Layout",
		entries: [{
			binding: 0,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
			buffer: {} // defaults to "type: uniform"
		},{
			binding: 1,
			visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
			buffer: {type: "read-only-storage"} 
		},{
			binding: 2,
			visibility: GPUShaderStage.COMPUTE,
			buffer: {type: "storage"}
		}]
	});

	const bindGroups = [
		device.createBindGroup({
			label: "Bind Group A",
			layout: bindGroupLayout,
			entries: [{
				binding: 0,
				resource: { buffer: uniformBuffer }
			},{
				binding: 1,
				resource: { buffer: cellStateBuffer[0] }
			},{
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
			},{
				binding: 1,
				resource: { buffer: cellStateBuffer[1] }
			},{
				binding: 2,
				resource: { buffer: cellStateBuffer[0] }
			}]
		})
	];

	const pipelineLayout = device.createPipelineLayout({
		label: "Cell Pipeline Layout",
		bindGroupLayouts: [ bindGroupLayout ]
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

	let step = 0;
	function UpdateTable() {
		const encoder = device.createCommandEncoder();
		// ComputePass
		const computePass = encoder.beginComputePass();
		
		computePass.setPipeline(simulationPipeline);
		computePass.setBindGroup(0, bindGroups[step % 2]);
		const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
		computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
		computePass.end();

		++step;
		// Clear canvas
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
		pass.draw(VERTEX_DATA.rect_vertices.length / 2, GRID_SIZE * GRID_SIZE);

		pass.end();
		device.queue.submit([encoder.finish()]); /* encoder.finish() returns the command buffer to submit */
	}

	setInterval(UpdateTable, UPDATE_INTERVAL);
}

try {
	start()
}
catch (e) {
	printError("start() - Uncaught JavaScript error.", $(e));
}