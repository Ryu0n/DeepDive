class SingletonInstance:
    __instance = None
    __tmp_method = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kwargs):
        cls.__instance = cls(*args, **kwargs)
        cls.__tmp_method = cls.instance
        cls.instance = cls.__getInstance
        return cls.__instance

    @classmethod
    def removeInstance(cls):
        if cls.__instance:
            cls.__instance = None
            cls.instance = cls.__tmp_method

    @classmethod
    def clearInstance(cls, *args, **kwargs):
        if cls.__instance:
            obj = cls.__new__(cls)
            obj.__init__(*args, **kwargs)
            cls.__instance = obj
            cls.instance = cls.__getInstance
            return cls.__instance