include $(MP_TOP)/py/mkenv.mk
CPP = $(CC) -E -x c
QSTR_DEFS = qstrdefsport.h
MICROPY_ROM_TEXT_COMPRESSION = 0
include $(TOP)/py/py.mk
INC += -I. -I../include -I$(TOP) -I$(BUILD)
CFLAGS += $(INC) $(RV_FLAGS) -std=gnu99 -DNDEBUG -fno-builtin -Os
CSUPEROPT = -Os
LDFLAGS += $(RV_FLAGS) -nostdlib -Wl,--gc-sections,--build-id=none -Wl,-T,../c_link.ld
SRC_C = main.c shared/libc/string0.c shared/libc/printf.c shared/runtime/gchelper_generic.c
SRC_QSTR += main.c
OBJ += $(PY_CORE_O) $(addprefix $(BUILD)/,$(SRC_C:.c=.o)) $(BUILD)/start.o $(BUILD)/guest_support.o
all: $(BUILD)/micropython.elf
$(BUILD)/start.o: ../c_start.S
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/guest_support.o: ../c_support.c
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/micropython.elf: $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^
include $(TOP)/py/mkrules.mk
