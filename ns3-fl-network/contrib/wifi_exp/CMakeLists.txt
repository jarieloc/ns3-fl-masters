set(source_files
        fl-client-application.cc
        fl-client-session.cc
        fl-energy.cc
        fl-experiment.cc
        fl-server.cc
        fl-server-helper.cc
        fl-sim-interface.cc
        main.cc
)

set(header_files
        fl-client-application.h
        fl-client-session.h
        fl-energy.h
        fl-experiment.h
        fl-server.h
        fl-server-helper.h
        fl-sim-interface.h
)

build_exec(
        EXECNAME wifi_exp
        SOURCE_FILES ${source_files}
  performance-simple-model.cc
  performance-model.cc
  power-model.cc
  temperature-model.cc
  reliability-model.cc
  cpu-energy-model.cc
        HEADER_FILES ${header_files}
        LIBRARIES_TO_LINK ${libcore} ${libenergy} ${libnetwork} ${libwifi} ${libinternet} ${libcsma}
        EXECUTABLE_DIRECTORY_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bin
)
