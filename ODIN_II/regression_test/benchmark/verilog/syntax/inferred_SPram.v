`define WORD_SIZE 32
`define DEPTH_LOG2 3
`define RAM_DEPTH 2**3

module inferred_spram(
	clk,

	address,
	data_in,
	data_out
);

	input clk;
	input [`DEPTH_LOG2-1:0] address;
	input [`WORD_SIZE-1:0] data_in;
	
	output [`WORD_SIZE-1:0] data_out;

	reg  [`WORD_SIZE-1:0] mregs [`RAM_DEPTH-1:0];  

	always @ (posedge clk )
	begin
		data_out <= data_in;
		mregs[address] <= data_in;
	end

endmodule
