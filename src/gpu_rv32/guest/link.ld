ENTRY(_start)
MEMORY { vm (rwx) : ORIGIN = 0, LENGTH = 64K }
SECTIONS {
    .text : { *(.text.start) *(.text .text.*) } > vm
    .rodata : ALIGN(16) { *(.rodata .rodata.*) *(.srodata .srodata.*) } > vm
    .data : ALIGN(16) { *(.data .data.*) *(.sdata .sdata.*) } > vm
    .bss (NOLOAD) : ALIGN(16) { *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON) } > vm
    ASSERT(. <= 32768, "interpreter image leaves less than 32 KiB stack")
    /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.eh_frame*) }
}
