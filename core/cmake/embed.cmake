# SPDX-License-Identifier: Apache-2.0
# cmake -DSRC=... -DDST=... -DSYMBOL=... -DMODE=bin|text -P embed.cmake
# bin : writes alignas(4) const unsigned char SYMBOL[] + SYMBOL_size
# text: writes const char* SYMBOL as a raw string literal
if(MODE STREQUAL "bin")
  file(READ "${SRC}" hex HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," arr "${hex}")
  file(SIZE "${SRC}" sz)
  file(WRITE "${DST}"
    "// generated from ${SRC}\nalignas(4) static const unsigned char ${SYMBOL}[] = {${arr}};\nstatic const unsigned long ${SYMBOL}_size = ${sz};\n")
elseif(MODE STREQUAL "text")
  file(READ "${SRC}" txt)
  file(WRITE "${DST}" "// generated from ${SRC}\nstatic const char* const ${SYMBOL} = R\"HPCSRC(${txt})HPCSRC\";\n")
else()
  message(FATAL_ERROR "embed.cmake: MODE must be bin or text")
endif()
