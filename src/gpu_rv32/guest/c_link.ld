ENTRY(_start)
MEMORY { vm (rwx) : ORIGIN = 0, LENGTH = 4M }
SECTIONS {
    .text : { *(.text.start) *(.text .text.*) } > vm
    .rodata : ALIGN(16) { *(.rodata .rodata.*) *(.srodata .srodata.*) } > vm
    .data : ALIGN(16) { *(.data .data.*) *(.sdata .sdata.*) } > vm
    .bss (NOLOAD) : ALIGN(16) { *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON) } > vm
    ASSERT(. <= 3932160, "image leaves less than 256 KiB stack")
    /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.eh_frame*) }
}
