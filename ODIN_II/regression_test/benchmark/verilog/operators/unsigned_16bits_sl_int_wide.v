module simple_op ( input [31:0] in,
                                output [31:0] out );

    assign out = in << 16;

endmodule
